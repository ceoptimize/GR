# %% [markdown]
# ## Neo4j Import of GraphRAG Result Parquet files
# This notebook imports the results of the GraphRAG indexing process into the Neo4j Graph database for further processing, analysis or visualization.
#
# ### How does it work?
# The notebook loads the parquet files from the output folder of your indexing process and loads them into Pandas dataframes. It then uses a batching approach to send a slice of the data into Neo4j to create nodes and relationships and add relevant properties. The id-arrays on most entities are turned into relationships.
#
# All operations use `MERGE`, so they are idempotent, and you can run the script multiple times.
#
# If you need to clean out the database, you can run the following statement
# ```
# MATCH (n)
# CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 25000 ROWS;
# ```

# %%
import time
from neo4j import GraphDatabase
import pandas as pd
GRAPHRAG_FOLDER = "artifacts"

# %%
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# %%
from neo4j import GraphDatabase

import os
#set up a variable called NEO4JACCESS which can be either AURA, LOCALWEB, or LOCALDESKTOP or DOCKER
NEO4JACCESS = "LOCALDESKTOP"

if NEO4JACCESS == "AURA":
    uri = os.getenv("NEO4J_URI_AURA")
    username = os.getenv("NEO4J_USERNAME_AURA")
    password = os.getenv("NEO4J_PASSWORD_AURA")
elif NEO4JACCESS == "LOCALWEB":
    #Todo: set up the local web instance
    pass
elif NEO4JACCESS == "LOCALDESKTOP":
    uri = os.getenv("NEO4J_URI_DESKTOP")
    username = os.getenv("NEO4J_USERNAME_DESKTOP")
    password = os.getenv("NEO4J_PASSWORD_DESKTOP")
elif NEO4JACCESS == "DOCKER":
    #Todo: set up the docker instance
    pass
else:
    pass

print(f"NEO4JACCESS is set to: {NEO4JACCESS}")
print(uri, username, password)


driver = GraphDatabase.driver(uri , auth=(username, password))

NEO4J_DATABASE = "christmascarol"


# %%
def batched_import(statement, df, batch_size=1000):
    """
    Import a dataframe into Neo4j using a batched approach.
    Parameters: statement is the Cypher query to execute, df is the dataframe to import, and batch_size is the number of rows to import in each batch.
    """
    total = len(df)
    start_s = time.time()
    for start in range(0, total, batch_size):
        batch = df.iloc[start: min(start+batch_size, total)]
        result = driver.execute_query("UNWIND $rows AS value " + statement,
                                      rows=batch.to_dict('records'),
                                      database_=NEO4J_DATABASE)
        print(result.summary.counters)
    print(f'{total} rows in { time.time() - start_s} s.')
    return total

# %% [markdown]
# ### Indexes and Constraints
# Indexes in Neo4j are only used to find the starting points for graph queries, e.g. quickly finding two nodes to connect. Constraints exist to avoid duplicates, we create them mostly on id's of Entity types.
#
# We use some Types as markers with two underscores before and after to distinguish them from the actual entity types.
#
# The default relationship type here is `RELATED` but we could also infer a real relationship-type from the description or the types of the start and end-nodes.
#
# * `__Entity__`
# * `__Document__`
# * `__Chunk__`
# * `__Community__`
# * `__Covariate__`

# %%
# create constraints, idempotent operation


statements = """
create constraint chunk_id if not exists for (c:__Chunk__) require c.id is unique;
create constraint document_id if not exists for (d:__Document__) require d.id is unique;
create constraint entity_id if not exists for (c:__Community__) require c.community is unique;
create constraint entity_id if not exists for (e:__Entity__) require e.id is unique;
create constraint entity_title if not exists for (e:__Entity__) require e.name is unique;
create constraint entity_title if not exists for (e:__Covariate__) require e.title is unique;
create constraint related_id if not exists for ()-[rel:RELATED]->() require rel.id is unique;
""".split(";")

for statement in statements:
    if len((statement or "").strip()) > 0:
        print(statement)
        driver.execute_query(statement)

# %% [markdown]
# ## Import Process
# ### Importing the Documents
# We're loading the parquet file for the documents and create nodes with their ids and add the title property. We don't need to store text_unit_ids as we can create the relationships and the text content is also contained in the chunks.

# %%
doc_df = pd.read_parquet(
    f'{GRAPHRAG_FOLDER}/create_final_documents.parquet', columns=["id", "title"])
doc_df.head(3)
#should only print one line

# %%
# import documents
statement = """
MERGE (d:__Document__ {id:value.id})
SET d += value {.title}
"""

batched_import(statement, doc_df)

# %% [markdown]
# ### Loading Text Units
# We load the text units, create a node per id and set the text and number of tokens. Then we connect them to the documents that we created before.

# %%
text_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_text_units.parquet',
                          columns=["id", "text", "n_tokens", "document_ids"])
text_df.head(2)

# %%
statement = """
MERGE (c:__Chunk__ {id:value.id})
SET c += value {.text, .n_tokens}
WITH c, value
UNWIND value.document_ids AS document
MATCH (d:__Document__ {id:document})
MERGE (c)-[:PART_OF]->(d)
"""

batched_import(statement, text_df)

# %% [markdown]
# ### Loading Nodes
# For the nodes we store id, name, description, embedding (if available), human readable id.

# %%
entity_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_entities.parquet',
                            columns=["name", "type", "description", "human_readable_id", "id", "description_embedding", "text_unit_ids"])
entity_df.head(2)

# %%
entity_statement = """
MERGE (e:__Entity__ {id:value.id})
SET e += value {.human_readable_id, .description, name:replace(value.name,'"','')}
WITH e, value
CALL db.create.setNodeVectorProperty(e, "description_embedding", value.description_embedding)
CALL apoc.create.addLabels(e, case when coalesce(value.type,"") = "" then [] else [apoc.text.upperCamelCase(replace(value.type,'"',''))] end) yield node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id:text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""

batched_import(entity_statement, entity_df)

# %% [markdown]
# ### Import Relationships
# For the relationships we find the source and target node by name, using the base `__Entity__` type. After creating the RELATED relationships, we set the description as attribute.

# %%
rel_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_relationships.parquet',
                         columns=["source", "target", "id", "rank", "weight", "human_readable_id", "description", "text_unit_ids"])
rel_df.head(2)

# %%
rel_statement = """
    MATCH (source:__Entity__ {name:replace(value.source,'"','')})
    MATCH (target:__Entity__ {name:replace(value.target,'"','')})
    // not necessary to merge on id as there is only one relationship per pair
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel += value {.rank, .weight, .human_readable_id, .description, .text_unit_ids}
    RETURN count(*) as createdRels
"""

batched_import(rel_statement, rel_df)

# %% [markdown]
# ### Importing Communities
# For communities we import their id, title, level. We connect the `__Community__` nodes to the start and end nodes of the relationships they refer to.
#
# Connecting them to the chunks they orignate from is optional, as the entites are already connected to the chunks.

# %%
community_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_communities.parquet',
                               columns=["id", "level", "title", "text_unit_ids", "relationship_ids"])

community_df.head(2)

# %%
statement = """
MERGE (c:__Community__ {community:value.id})
SET c += value {.level, .title}
/*
UNWIND value.text_unit_ids as text_unit_id
MATCH (t:__Chunk__ {id:text_unit_id})
MERGE (c)-[:HAS_CHUNK]->(t)
WITH distinct c, value
*/
WITH *
UNWIND value.relationship_ids as rel_id
MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURN count(distinct c) as createdCommunities
"""

batched_import(statement, community_df)

# %% [markdown]
# ### Importing Community Reports
# Fo the community reports we create nodes for each communitiy set the id, community, level, title, summary, rank, and rank_explanation and connect them to the entities they are about. For the findings we create the findings in context of the communities.

# %%
community_report_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_community_reports.parquet',
                                      columns=["id", "community", "level", "title", "summary", "findings", "rank", "rank_explanation", "full_content"])
community_report_df.head(2)

# %%
# import communities
community_statement = """
MERGE (c:__Community__ {community:value.community})
SET c += value {.level, .title, .rank, .rank_explanation, .full_content, .summary}
WITH c, value
UNWIND range(0, size(value.findings)-1) AS finding_idx
WITH c, value, finding_idx, value.findings[finding_idx] as finding
MERGE (c)-[:HAS_FINDING]->(f:Finding {id:finding_idx})
SET f += finding
"""
batched_import(community_statement, community_report_df)

# %% [markdown]
# ### Importing Covariates
# Covariates are for instance claims on entities, we connect them to the chunks where they originate from.
#
# **By default, covariates are not included in the output, so the file might not exists in your output if you didn't set the configuration to extract claims**

# %%
"""
# cov_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_covariates.parquet')
# cov_df.head(2)
"""

# %%
cov_statement = """
MERGE (c:__Covariate__ {id:value.id})
SET c += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
WITH c, value
MATCH (ch:__Chunk__ {id: value.text_unit_id})
MERGE (ch)-[:HAS_COVARIATE]->(c)
"""
# batched_import(cov_statement, cov_df)

# %%
print(uri, username, password)
# %%
#print(result.summary.counters)
# %%
doc_df.head()         # For documents
# %%
text_df.head()        # For chunks/text units
#%%
entity_df.head()      # For entities
# %%
rel_df.head()         # For relationships
# %%
community_df.head()   # For communities
#  %%
community_report_df.head()  # For community reports
# %%
