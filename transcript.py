import json
import requests
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import os
import random
import sys
import time
import itertools
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

#Load Call Transcript Data
#sample data period
bgn_dt = 20251001
end_dt = 20251231

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_path = os.path.join("/Workspace", *notebook_path.split("/")[0:-2])
print("project path:", project_path)

telephone_df = spark.read.table("sample_transacript_db.telephonecommunication")\
  .select("TelephoneCommunicationId", "Language", "DataDate")\
  .filter(F.col("DataDate").between(bgn_dt, end_dt))\
  .withColumnRenamed("TelephoneCommunicationId", "ContactId")

transcript_masked_df = spark.read.table("sample_transacript_db_lab.telephonecommunication_masked")\
  .select("TelephoneCommunicationId", "SequencedTranscript_masked")\
  .filter(F.col("DataDate").between(bgn_dt, end_dt))\
  .withColumnRenamed("TelephoneCommunicationId", "ContactId")\
  .withColumnRenamed("SequencedTranscript_masked", "TranscriptMasked")

transcript_df = transcript_masked_df.join(telephone_df, on="ContactId", how="Inner")\
  .select("DataDate","ContactId", "Language", "TranscriptMasked")\
  .filter(F.col("TranscriptMasked").isNotNull())\
  .filter(F.col("Language") == "English")\
  .orderBy("DataDate", "ContactId")\
  .persist(StorageLevel.MEMORY_AND_DISK)


n_total = transcript_df.count()
print(f"Total: {n_total}")

# Record count sense check
df = transcript_df\
    .groupBy("DataDate").count()\
    .withColumn("DataDate_", F.to_date(F.col("DataDate").cast("string"), "yyyyMMdd"))\
    .withColumn("Weekday", F.date_format(F.col("DataDate_"), "E"))\
    .select("DataDate", "Weekday", "count")\
    .orderBy(F.col("DataDate"))
        
display(df)

# Sampling, to avoid too much topics that exceed context window of LLM
n_sample = 10000
transcript_df = transcript_df.sample(withReplacement=False, fraction=1.0, seed=1234).limit(n_sample)

print(f"Sampled: {transcript_df.count()} out of {n_total} ({round(100 * transcript_df.count() / n_total, 2)}%)")
     
#Configure Azure OpenAI


# Azure AD details
tenant_id = 'azure_spn_tenant_id'
client_id = 'azure_spn_client_id'
client_secret = dbutils.secrets.get(scope = f"key-vault-secrets-allusers",key='KV-Translate-SPN-Secret')
print(client_secret)
scope = "https://cognitiveservices.azure.com/.default"

# Obtain an access token
token_url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
token_data = {
'grant_type': 'client_credentials',
'client_id': client_id,
'client_secret': client_secret,
'scope': scope
}


token = ""
headers = {}
def refresh_token():
    global token, headers
    token_r = requests.post(token_url, data=token_data)
    token = token_r.json().get("access_token")
    headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': f'application/json',
    }

refresh_token()

#Generate GenAI Topics

system = \
"""You are a life insurance customer service contact center expert, and your role is to extract the customer's topics of discussion from a call transcript in //LANGUAGE// between a call center servicing Agent and a Customer, along with a short summary explaining customer's call intent. The dialog is marked by "CUSTOMER:" and "AGENT:". A call can have multiple topics.

Instructions
1. Exclude any PII (Personally Identifiable Information) such as customer name, policy number, agent name, address, email address, and phone number from the output.
2. Ensure the output language is in //LANGUAGE//.
3. Generate a short summary explaining the customer's call intent (up to 300 words).
4. Generate business-relevant topics.
    4.1. Maintain logical progression and coherence in insurance-related knowledge across three levels:
       - Level-1 is the highest-level topics
       - Level-2 is the intermediate concepts
       - Level-3 is the the advanced insights
    4.2 Verify that each level builds upon the previous one, ensuring clarity and a sensible flow of information.
5. Attend to following constraints when generating topics.
    4.3 Important: Limit strictly to Level-1 categories that are provided below. Do not create new Level-1 topics.
    4.4 Important: Elaborate on each of the Level-1 categories, breaking every of them down comprehensively to reach Level-3 sub-categories.
    4.5 Important: Avoid using any provided Level-1 categories within Level-2 and Level-3 sub-categories.
5. Consider the following insurance context when identifying and creating topics.
    5.1 Only include claim-related topics if and only if there is an explicit and clear indication of an actual or potential insurance claim. This includes situations where the customer is filing, following up, or inquiring about a claim, and discussions on reimbursement, claim status, or related issues. An insurance claim is a formal request for compensation for a loss or event covered under the policy. Differentiate the specific term 'insurance claim' with the general use of the word 'claim', which means to state or assert something.
    5.2 Distinguish underwriting-related topics, which pertain to the application, evaluation and approval of insurance policies.
    5.3 Distinguish surrender or withdrawal-related topics, which involve canceling a policy and withdrawing its cash value, or making partial withdrawals from the policy.
    5.4 Distinguish between general policy application and policy loan application. A policy application initiates a new insurance policy, involving underwriting and coverage terms, whereas a policy loan involves borrowing against a life insurance policy's cash value using it as collateral.
    5.5 Distinguish between premium-related topics and loan-related topics. Premiums are regular payments made to keep an insurance policy active, whereas loans usually refer to borrowed amounts that need to be repaid.


Here are all the possible options of Level-1 categories:
- App
- Call
- Change Address or Telephone or Email
- Change Billing Mode or Method
- Claim
- Cost/Fees
- Coupon
- Coverage
- Customer Service Representative
- Feature Benefits
- Giro
- Maturity
- Nob Assignment
- Policy Loan
- Premium
- Reinstatement
- Surrender Termination
- Website


Here are the examples of topics (in hierarchical format of 'Level-1 | Level-2 | Level-3').
Important: these examples are incomplete and may only include Level-1 and Level-2 categories. Your task is to elaborate and break each of them down comprehensively to reach Level-3 sub-categories.
- App | Authentication | Changing/Updating Password
- Call | Ivr | Phone Menu
- Change Address or Telephone or Email | Update Address Manually
- Change Billing Mode or Method | Billing Method
- Claim | Documentation Requirement
- Cost/Fees | Fees | Account Closure Fee
- Coupon | Accumulate Payout Offset
- Coverage | Product Policy | Policy Information
- Customer Service Representative | Issue Resolution | Effectiveness Of Resolution
- Feature Benefits | Payout Period
- Giro | Giro Deduction Status
- Maturity | Maturity Date
- Nob Assignment | Beneficiary Names
- Policy Loan | Maximum Loan Amount
- Premium | Premium Freeze
- Reinstatement | Reinstate Policy
- Surrender Termination | Surrender Value
- Website | Authentication | Login/Authentication


#### Output Format
Output must be in JSON format and in //LANGUAGE//. The JSON object should contain the following keys:
{
    "short_summary": "max 300 words short summary explaining customer's call intent",

    "topics_generated": [
        # Business-relevant topics which all Level-1, Level-2, and Level-3 are aligned consistently and have clear relevance to the 'short_summary'
        # Ensure topics output in the format 'Level-1 | Level-2 | Level-3'
        # Always use singular instead of plural word forms
    ]
}"""
 
refresh_token()

api_url = "open_ai_url"

def predict(transcript, language):

    global token, headers

    max_attempt = 3
    seed = 42
    
    for attempt in range(1, max_attempt + 1):

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system.replace("//LANGUAGE//", language)
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 1000
        }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response_text = response.text.replace("```json", "").replace("```", "") #fix occasional wrong json format
            response_content = json.loads(response_text)['choices'][0]['message']['content']
            response_content = json.loads(response_content)
            response_content["status_code"] = str(response.status_code)
            return response_content
        except json.JSONDecodeError as e:
            if attempt >= max_attempt:
                return {"statuscode": "JSONDecodeError" + ": " + str(e)}
            else:
                seed += 1
                continue
        except Exception as e:
            if attempt >= max_attempt:
                return {"statuscode": str(response.status_code) + ' '+ str(e)}
            elif response.status_code == 429: #Too Many Requests
                # Exponential backoff with jitter
                sleep_time = 60 * (3 ** (attempt - 1)) #60, 180, 540 seconds
                jitter = random.uniform(0, sleep_time * 0.1)  # Add up to 10% jitter
                time.sleep(sleep_time + jitter)
                continue
            elif response.status_code == 401: #Unauthorized. Access token is missing, invalid, audience is incorrect
                refresh_token()
                continue
            else:
                #400 Unable to parse and estimate tokens from incoming request
                time.sleep(3)
                continue

returnType = T.StructType([
    T.StructField("status_code", T.StringType()),
    T.StructField("short_summary", T.StringType()),
    T.StructField("topics_generated", T.ArrayType(T.StringType())),
])

predict_udf = udf(predict, returnType=returnType)

prediction_df = transcript_df\
                .repartition(8)\
                .fillna({"TranscriptMasked": "Null"})\
                .withColumn("Prediction", predict_udf(F.col("TranscriptMasked"), F.col("Language")))

prediction_df = prediction_df\
              .withColumn("StatusCode", F.col("Prediction.status_code"))\
              .withColumn("ShortSummary", F.col("Prediction.short_summary"))\
              .withColumn("TopicsGenerated", F.col("Prediction.topics_generated"))\
              .orderBy("DataDate", "ContactId")\
              .drop("Prediction")\
              .persist(StorageLevel.MEMORY_AND_DISK)

# display(prediction_df)

print("API calls succeeded:", prediction_df.filter(F.col("StatusCode")=="200").count(), "out of", prediction_df.count())

path = "abfss://lab@azcontainer.dfs.core.windows.net/project/customer_service_data/CAR_Topic_modelling_data/GenAI_Topic_List/topics_generated"
prediction_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
   
   
topics_level1 = [
    "App",
    "Call",
    "Change Address or Telephone or Email",
    "Change Billing Mode or Method",
    "Claim",
    "Cost/Fees",
    "Coupon",
    "Coverage",
    "Customer Service Representative",
    "Feature Benefits",
    "Giro",
    "Maturity",
    "Nob Assignment",
    "Policy Loan",
    "Premium",
    "Reinstatement",
    "Surrender Termination",
    "Website",
]


prediction_df = spark.read.load("abfss://lab@azcontainer.dfs.core.windows.net/project/customer_service_data/modelling_data/topics_generated")


exploded_df = prediction_df\
                .select("ContactID", "Language", "TopicsGenerated")\
                .withColumn("TopicsGenerated", F.explode(F.col("TopicsGenerated")))\
                .groupBy(["Language", "TopicsGenerated"]).agg(F.count("TopicsGenerated").alias("Count"))\
                .withColumn("TopicsLevel1", F.trim(F.split(F.col("TopicsGenerated"), "\|").getItem(0)))\
                .withColumn("TopicsLevel2", F.trim(F.split(F.col("TopicsGenerated"), "\|").getItem(1)))\
                .withColumn("TopicsLevel3", F.trim(F.split(F.col("TopicsGenerated"), "\|").getItem(2)))\
                .orderBy(F.asc("TopicsGenerated"))


#keep only topics with certain count or above. Put '1' if no filtering
#exclude topics which do not develop down to Level-3
window_spec = Window.orderBy("TopicsGenerated") #.partitionBy("TopicsLevel1")
exploded_df = exploded_df\
                .filter(F.col("Count") >= 2)\
                .filter(F.col("TopicsLevel1").isin(topics_level1))\
                .filter(F.col("TopicsLevel2").isNotNull() & F.col("TopicsLevel3").isNotNull())\


exploded_df = exploded_df\
                .withColumn("Index", F.row_number().over(window_spec))\
                .withColumn("Index", F.array(F.col("Index")))\
                .select("Index", "Language", "TopicsLevel1", "TopicsLevel2", "TopicsLevel3", "TopicsGenerated", "Count")

print(exploded_df.count())

display(exploded_df)

#Consolidate GenAI Generated Topics Deduplication

system = \
"""You are a life insurance customer service contact centre expert and your role is to deduplicate and consolidate a list of customer's topics of discussion into a more compact and well-structured list. Each topic is assigned an index number and presented in hierarchical format of 'Level-1 | Level-2 | Level-3'. Your goal is to identify overlapping themes and combine very similar or closely related items under broader categories, where applicable. Merge topics with similar meanings but different wording to create a cohesive list.

Instructions:
1. Review the entire list of topics, identifying similar or redundant entries, and consolidate them where possible.
2. Assign indexes exhaustively and exclusively to each consolidated topic.
3. Output only the indexes, and omit any topic descriptions.


Few-shot Example #1 -

[11] Surrender Termination | Documentation Requirement | Necessary Documents for Surrender
[12] Surrender Termination | Documentation Requirement | Required Documents for Surrender
[13] Surrender Termination | Documentation Requirement | Required Documents

Output-
{"index_consolidated": [[11,12,13]]}


Few-shot Example #2 -

[41] Claim | Claim Process | Steps to File a Claim
[42] Claim | Claim Process | Submission of Claims
[44] Claim | Submission Process | Claim Submission
[47] Claim | Claim Process | Claim Submission Requirements
[51] Claim | Appeal Process | Submission of Appeal
[52] Claim | Appeal Process | Submission Guidelines

Output-
{"index_consolidated": [[41,42,44,47],[51,52]]}

Few-shot Example #3 -

[71] Maturity | Maturity Date | Estimated Payout Value
[72] Maturity | Maturity Date | Policy Expiration
[74] Maturity | Maturity Value | Projected Maturity Amount
[75] Maturity | Maturity Date | Policy End Date

Output-
{"index_consolidated": [[71, 74],[72, 75]]}



#### Output Format
Output must be in JSON format. The JSON object should contain the following keys:
{
    "index_consolidated": [[# List of index numbers]]
        # Example: [[11,12,13],[41,42,44,47],[51,52],[71, 74],[72, 75]] 
}

"""
refresh_token()

api_url = "open_ai_url"

def deduplicate(topics_list):

    max_attempt = 3
    seed = 42
    
    for attempt in range(1, max_attempt + 1):

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": topics_list
                }
            ],
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 16384
        }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response_text = response.text.replace("```json", "").replace("```", "") #fix occasional wrong json format
            response_content = json.loads(response_text)['choices'][0]['message']['content']
            response_content = json.loads(response_content)
            response_content["status_code"] = str(response.status_code)
            return response_content
        except:
            if attempt >= max_attempt:
                error_topic = topics_list.split('\n')[0]
                raise Exception(f"Error: Maximum attempts exceeded on topic list: {error_topic} ...")
            elif response.status_code == 429: #Too Many Requests
                # Exponential backoff with jitter
                sleep_time = 60 * (3 ** (attempt - 1)) #60, 180, 540 seconds
                jitter = random.uniform(0, sleep_time * 0.1)  # Add up to 10% jitter
                time.sleep(sleep_time + jitter)
                continue
            elif response.status_code == 401: #Unauthorized. Access token is missing, invalid, audience is incorrect
                refresh_token()
                continue
            else:
                seed += 1
                topics_list += "\" #workaround for 'filtered due to the prompt triggering Azure OpenAI's content management policy'
                continue

    return response_content

returnType = T.StructType([
    T.StructField("status_code", T.StringType()),
    T.StructField("index_consolidated", T.ArrayType(T.ArrayType(T.LongType())))
])


deduplicate_udf = udf(deduplicate, returnType=returnType)
     
dedup_df = exploded_df\
            .select("Index", "TopicsLevel1", "TopicsGenerated")\
            .withColumn("Index", F.expr("concat('[', array_join(Index, ', '), ']')"))\
            .withColumn("TopicsList", F.concat(F.col("Index"), F.lit(" "), F.col("TopicsGenerated")))\
            .drop("Index", "TopicsGenerated")

dedup_df = dedup_df\
            .orderBy(F.asc("TopicsGenerated"))\
            .groupBy("TopicsLevel1").agg(F.concat_ws("\n", F.collect_list("TopicsList")).alias("TopicsList"))


print(dedup_df.count())

dedup_df = dedup_df\
            .repartition(8)\
            .withColumn("Prediction", deduplicate_udf(F.col("TopicsList")))

dedup_df = dedup_df\
            .withColumn("StatusCode", F.col("Prediction.status_code"))\
            .withColumn("IndexConsolidated", F.col("Prediction.index_consolidated"))\
            .orderBy(F.asc("TopicsLevel1"))\
            .drop("Prediction")\
            .persist(StorageLevel.MEMORY_AND_DISK)

display(dedup_df)
print("API calls succeeded:", dedup_df.filter(F.col("StatusCode")=="200").count(), "out of", dedup_df.count())


#Consolidation

system = \
"""You are a life insurance customer service contact centre expert and your role is to summarize and generalize a list of customer's topics of discussion into one single topic. Each topic is presented in hierarchical format of 'Level-1 | Level-2 | Level-3'.

Instructions:
1. Ensure the output language is in //LANGUAGE//.
2. Organize the topics into a hierarchy with clear categories and subcategories.
3. Ensure that each consolidated topic reflects the core idea of the original topics it encompasses.
4. Present the final list of topics in a clean, readable format with clear hierarchy levels.
5. Always use singular instead of plural word forms for consolidated topics.
6. Write a description in //LANGUAGE// for each consolidated entry (up to 30 words). Instead of merely rephrasing the words in the topics, elaborate with additional information (e.g., expanding abbreviations) to facilitate readers' understanding. Be formal and objective.


Few-shot Example #1 -

Surrender Termination | Documentation Requirement | Necessary Document for Surrender
Surrender Termination | Documentation Requirement | Required Document for Surrender
Surrender Termination | Documentation Requirement | Required Document

Output:
{"topic_consolidated": "Surrender Termination | Documentation Requirement | Required Document", "topics_description": "Overview of necessary documentation for surrender termination, detailing essential documents needed to process and finalize a surrender."}


Few-shot Example #2 -

Claim | Claim Process | Steps to File a Claim
Claim | Claim Process | Submission of Claims
Claim | Submission Process | Claim Submission
Claim | Claim Process | Claim Submission Requirements

Output-
{"topic_consolidated": "Claim | Claim Process | Claim Submission", "topics_description": "Submission steps, requirements, and procedures for filing claims."}


Few-shot Example #3 -

Maturity | Maturity Date | Estimated Payout Value
Maturity | Maturity Value | Projected Maturity Amount

Output-
{"topic_consolidated": "Maturity | Maturity Value | Payout Amount Assessment", "topics_description": "Evaluation and estimation of maturity values and amounts related to life insurance policies."}


#### Output Format
Output must be in JSON format and in //LANGUAGE//. The JSON object should contain the following keys:
{
    "topic_consolidated": "Consolidated topic in the format 'Level-1 | Level-2 | Level-3'",
    "topic_description": "max 30 words description summarizing the provided topics"
}

"""

refresh_token()

api_url = "open_ai_url"

def consolidate(topics_list, language):

    max_attempt = 10
    seed = 42
    
    for attempt in range(1, max_attempt + 1):

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system.replace("//LANGUAGE//", language)
                },
                {
                    "role": "user",
                    "content": topics_list
                }
            ],
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 16384
        }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response_text = response.text.replace("```json", "").replace("```", "") #fix occasional wrong json format
            response_content = json.loads(response_text)['choices'][0]['message']['content']
            response_content = json.loads(response_content)
            response_content["status_code"] = str(response.status_code)
            return response_content
        except:
            if attempt >= max_attempt:
                error_topic = topics_list.split('\n')[0]
                raise Exception(f"Error: Maximum attempts exceeded on topic list: {error_topic} ...")
            elif response.status_code == 429: #Too Many Requests
                # Exponential backoff with jitter
                sleep_time = 60 * (3 ** (attempt - 1)) #60, 180, 540 seconds
                jitter = random.uniform(0, sleep_time * 0.1)  # Add up to 10% jitter
                time.sleep(sleep_time + jitter)
                continue
            elif response.status_code == 401: #Unauthorized. Access token is missing, invalid, audience is incorrect
                refresh_token()
                continue
            else:
                seed += 1
                topics_list += "\" #workaround for 'filtered due to the prompt triggering Azure OpenAI's content management policy'
                print(topics_list)
                continue

    return response_content

returnType = T.StructType([
    T.StructField("status_code", T.StringType()),
    T.StructField("topic_consolidated", T.StringType()),
    T.StructField("topic_description", T.StringType())
])


consolidate_udf = udf(consolidate, returnType=returnType)

conso_df = dedup_df\
            .select(F.explode(F.col("IndexConsolidated")).alias("IndexConsolidated"))\
            .select("IndexConsolidated", F.explode(F.col("IndexConsolidated")).alias("Index"))\
            .withColumn("Index", F.array(F.col("Index")))
            
conso_df = conso_df.join(exploded_df, on="Index", how="inner")\
            .select("IndexConsolidated", "Index", "Language", "TopicsGenerated", "Count")

conso_df = conso_df.groupBy("Language", "IndexConsolidated").agg(
            F.concat_ws("\n", F.collect_list("TopicsGenerated")).alias("TopicsList"),
            F.sum("Count").alias("Count")
            )


print(conso_df.count())

# display(conso_df)

display(conso_df)


conso_df = conso_df\
            .repartition(8)\
            .withColumn("Prediction", consolidate_udf(F.col("TopicsList"), F.col("Language")))

conso_df = conso_df\
            .withColumn("StatusCode", F.col("Prediction.status_code"))\
            .withColumn("TopicConsolidated", F.col("Prediction.topic_consolidated"))\
            .withColumn("TopicDescription", F.col("Prediction.topic_description"))\
            .orderBy(F.asc("TopicConsolidated"))\
            .drop("Prediction")\
            .persist(StorageLevel.MEMORY_AND_DISK)


display(conso_df)
print("API calls succeeded:", conso_df.filter(F.col("StatusCode")=="200").count(), "out of", conso_df.count())

path = "abfss://lab@azcontainer.dfs.core.windows.net/project/customer_service_data/modelling_data/topics_consolidated"
conso_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    
#Finalize List of Topics

select_df = conso_df\
            .withColumnRenamed("TopicConsolidated", "Topic")\
            .withColumnRenamed("TopicDescription", "Description")\
            .withColumn("Level1", F.trim(F.split(F.col("Topic"), "\|").getItem(0)))\
            .withColumn("Level2", F.trim(F.split(F.col("Topic"), "\|").getItem(1)))\
            .withColumn("Level3", F.trim(F.split(F.col("Topic"), "\|").getItem(2)))\
            .select("Language", "Level1", "Level2", "Level3", "Topic", "Description", "Count")\


#shorten the list of topics by excluding topics with low occurence
max_topics = 300
for threshold in range(1, 100 + 1): 
    select_df = select_df\
                .filter(F.col("Count") >= threshold)\
                .dropDuplicates(["Topic"])\
                .orderBy("Topic")
    if select_df.count() <= max_topics:
        break

print("threshold:", threshold)
display(select_df)

path = "abfss://lab@azcontainer.dfs.core.windows.net/project/customer_service_data/modelling_data/topics_finalized"
select_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)

with pd.ExcelWriter("/Workspace/Users/Topic Generation/List of Topics.xlsx") as writer:
    select_df.toPandas().to_excel(writer, sheet_name='List of Topics', index=False)
    #conso_df.toPandas().to_excel(writer, sheet_name='Raw Topics Pre Consolidation', index=False)
     
json_list = select_df \
    .select("Topic", "Description") \
    .withColumnRenamed("Topic", "tag") \
    .withColumnRenamed("Description", "description") \
    .toPandas().to_dict(orient='records')

with open("/Workspace/Users/Topic Generation/List of Topics.txt", "w") as file:

    file.write("[\n")

    length = len(json_list)
    for i, element in enumerate(json_list):
        json_string = json.dumps(element, ensure_ascii=False)
        
        file.write(json_string)
        if i < length - 1:
            file.write(',\n')

    file.write("\n]")
