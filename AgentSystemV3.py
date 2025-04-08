from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from prompt import role_prompt, your_name, role_name, summary_knowledge_prompt, summary_memory_prompt, stat_prompt, task_prompt
from neo4j import GraphDatabase, NotificationCategory



class AgentSystem:
  
  def __init__(self):
    self.api_key = API_KEY
    self.base_url = "https://api.deepseek.com"
    self.model = "deepseek-chat"

    self.chat_content_list = []
    self.memory_dict = {}
    self.relation_dict = {}
    self.knowledge_dict = {}
    self.respond = None
    self.system_prompt = None

    self.memory_engine = MemoryEngine()
    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


  def end_chat(self):
    '''手动结束对话, 让系统开始总结对话结果, 并保存在数据库中'''
    if self.system_prompt != None:
      print("Enter Summary")
      self.memory_engine.summary_chat(
        self.knowledge_dict,
        self.relation_dict,
        self.memory_dict,
        self.chat_content_list
      )
      
    self.knowledge_dict.clear()
    self.relation_dict.clear()
    self.memory_dict.clear()
    self.chat_content_list.clear()
    self.system_prompt = None
    self.respond = None

  def chat(self, new_text=None):
    if new_text != None:
      self.update_dict_state(new_text)
      self.chat_content_list.append({
        "role": your_name,
        "content": new_text
      })
    self.fetch_message()

  def fetch_message(self):
    '''向api发送输入请求回复'''
    messages = self.process_messages()
    self.response = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      temperature=1.5,
      stream=False
    )

    new_text = self.response.choices[0].message.content
    self.chat_content_list.append({
      "role": role_name,
      "content": new_text
      })
    print(new_text)
    self.update_dict_state(new_text)

  def update_dict_state(self, new_text):
    new_memory_dict, mem_connect_knowledge_dict = self.memory_engine.get_memory(new_text)
    knowledge_dict, relation_dict = self.memory_engine.get_knowledge(new_text)

    self.memory_dict.update(new_memory_dict)
    self.knowledge_dict.update(mem_connect_knowledge_dict)
    self.knowledge_dict.update(knowledge_dict)
    self.relation_dict.update(relation_dict)

  def process_state_prompt(self):
    state_prompt = []
    for id, memory in self.memory_dict.items():
      format_memory = f"id: {id} 记忆描述: {memory['description']}"
      state_prompt.append(format_memory)

    for id, knowledge in self.knowledge_dict.items():
      format_knowledge = f"id: {id} 名称: {knowledge['name']} 描述: {knowledge['description']}"
      state_prompt.append(format_knowledge)

    for id, relation in self.relation_dict.items():
      format_relation = f"{relation['from']}与{relation['to']}: {relation['description']}"
      state_prompt.append(format_relation)
    return "\n".join(state_prompt)
  
  def process_chat_content_list_messages(self):
    chat_content_list_messages = []
    for item in self.chat_content_list:
      chat_content_list_messages.append({
        "role": "user" if item["role"] == your_name else "assistant",
        "content": item["content"]
      })
    return chat_content_list_messages
  
  def process_messages(self):
    if self.system_prompt == None:
      self.system_prompt = self.memory_engine.get_system_prompt()
      self.memory_dict.update(self.memory_engine.get_init_memory())

    state_prompt = self.process_state_prompt()
    chat_content_list_messages = self.process_chat_content_list_messages()
    messages = [{"role": "system", "content": self.system_prompt + state_prompt}] + chat_content_list_messages
    return messages

class MemoryEngine:
  def __init__(self):
    self.model = "deepseek-chat"
    self.embed_model = "./multilingual-e5-large-instruct"
    self.api_key = API_KEY
    self.base_url = "https://api.deepseek.com"

    self.encoder = SentenceTransformer(self.embed_model)
    self.summary_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    self.dataset_init()
  
  def dataset_init(self):
    URL = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = ""
    self.driver = GraphDatabase.driver(URL, auth=(USER, PASSWORD),
      notifications_disabled_categories=[NotificationCategory.DEPRECATION])
    
    self.initialize_indexes()

  def initialize_indexes(self):
    with self.driver.session() as session:
      index_queries = [
        """
        create vector index memory_embedding if not exists 
        for (n:Memory) on (n.embedding) 
        options { indexConfig: { `vector.dimensions`: 1024, `vector.similarity_function`: 'cosine' } }
        """,
        """
        create vector index knowledge_embedding if not exists 
        for (n:Knowledge) on (n.embedding) 
        options { indexConfig: { `vector.dimensions`: 1024, `vector.similarity_function`: 'cosine' } }
        """,
        "CREATE INDEX knowledge_name IF NOT EXISTS FOR (k:Knowledge) ON (k.name)"
      ]

      for query in index_queries:
        session.execute_write(lambda tx: tx.run(query))

      print("Indexes and constraints initialized.")

  def memories_encoder(self, texts: list[str]):
    '''基于提供的文本, 返回其编码结果, 用于进行查找或索引'''
    return self.encoder.encode(texts, precision='float32')
  
  def get_init_memory(self):
    init_memory_query = """
      OPTIONAL MATCH (now:Dialog)
      WHERE NOT (now)-[:NEXT]->()
      WITH now
      OPTIONAL MATCH (now)-[:HAS_MEMORY]->(memory: Memory)
      RETURN
        ID(memory) AS memory_id,
        memory.name AS memeory_name,
        memory.description AS memory_description
    """

    with self.driver.session() as session:
      init_memory_item = session.execute_read(lambda tx: tx.run(init_memory_query).data())

    memory_dict = {}
    for record in init_memory_item:
      if record["memory_id"] != None:
        memory_dict[record["memory_id"]] = {
          "name": record["memeory_name"],
          "description": record["memory_description"]
        }
    return memory_dict


  def get_system_prompt(self):
    dialog_query = """
      OPTIONAL MATCH (now:Dialog)
      WHERE NOT (now)-[:NEXT]->()
      RETURN 
        now.plan as plan,
        now.state as state,
        now.task as task
    """
    with self.driver.session() as session:
      record = session.execute_read(lambda tx: tx.run(dialog_query).single())

    if record and record["state"] != None:
      prompt = "\n".join([
        "[状态仪表盘]",
        record["state"],
        "[角色任务]",
        record["task"],
        "[角色计划]",
        record["plan"],
      ])
    else:
      prompt = role_prompt +"\n"+ stat_prompt + "\n" + task_prompt
    return prompt  

  def get_memory(self, text:str):
    def get_detailed_instruct(task_description: str, query: str) -> str:
      return f'Instruct: {task_description}\nQuery: {query}'
    
    embedding = self.memories_encoder([
      get_detailed_instruct(
        "给定一段对话内容, 寻找与说话人所说的话的相关的记忆.", 
        text
      )
    ])[0]

    memory_query = """
      CALL db.index.vector.queryNodes('memory_embedding', 3, $embedding)
      YIELD node AS memory, score
      WHERE score > 0.94

      WITH memory, score
      OPTIONAL MATCH (memory)-[:HAS_KNOWLEDGE]->(knowledge: Knowledge)
      WITH DISTINCT memory, knowledge

      RETURN
        ID(memory) AS memory_id,
        memory.description AS memory_description,
        memory.name AS memory_name,
        ID(knowledge) AS knowledge_id,
        knowledge.name AS knowledge_name,
        knowledge.description AS knowledge_description
    """

    with self.driver.session() as session:
      memory_search_result = session.execute_read(
        lambda tx: tx.run(memory_query, embedding=embedding.tolist()).data()
      )
    
    memory_dict = {}
    knowledge_dict = {}

    for record in memory_search_result:
      memory_id = record['memory_id']
      if memory_id != None and memory_id not in memory_dict:
        memory_dict[memory_id] = {
          'name': record['memory_name'],
          'description': record['memory_description'],
        }
      
      knowledge_id = record['knowledge_id']
      if knowledge_id != None and knowledge_id not in knowledge_dict:
        knowledge_dict[knowledge_id] = {
          "name": record['knowledge_name'],
          "description": record['knowledge_description']
        }

    return memory_dict, knowledge_dict


  def get_knowledge(self, text:str):

    def get_detailed_instruct(task_description: str, query: str) -> str:
      return f'Instruct: {task_description}\nQuery: {query}'
    
    embedding = self.memories_encoder([
      get_detailed_instruct(
        "给定一句, 寻找话中提及的事物的相关的信息.", 
        text
      )
    ])[0]

    knowledge_query = """
      CALL db.index.vector.queryNodes('knowledge_embedding', 3, $embedding)
      YIELD node AS knowledge, score
      WHERE score > 0.92

      WITH collect(knowledge) AS knowledgeNodes

      UNWIND knowledgeNodes AS kn
      OPTIONAL MATCH (kn)-[r:RELATE_KNOWLEDGE]-(neighbor:Knowledge)
      WITH DISTINCT kn, neighbor, r
      RETURN 
        ID(kn) AS from_id,
        kn.name AS from_name,
        kn.description AS from_description,
        ID(neighbor) AS to_id,
        neighbor.name AS to_name,
        neighbor.description AS to_description,
        ID(r) AS relation_id,
        r.description AS relation_description
    """
    with self.driver.session() as session:
      knowledge_search_result = session.execute_read(
        lambda tx: tx.run(knowledge_query, embedding=embedding.tolist()).data()
      )

    knowledge_dict = {}
    relation_dict = {}

    for record in knowledge_search_result:
      from_id = record["from_id"]
      from_name = record["from_name"]
      from_desc = record["from_description"]
      if from_id not in knowledge_dict:
        knowledge_dict[from_id] = {
          "name": from_name,
          "description": from_desc
        }

      to_id = record["to_id"]
      to_name = record["to_name"]
      to_desc = record["to_description"]
      if to_id is not None and to_id not in knowledge_dict:
        knowledge_dict[to_id] = {
          "name": to_name,
          "description": to_desc
        }

      relation_desc = record["relation_description"]
      relation_id = record["relation_id"]
      if to_id is not None and relation_desc is not None:
        relation_dict[relation_id]= {
          "from": from_name,
          "to": to_name,
          "description": relation_desc,
        }
    return knowledge_dict, relation_dict

  def summary_chat(self, knowledge_dict, relation_dict, memory_dict, chat_content_list):
    '''
    简而言之, 对于记忆系统, 应该生成一个更加完备的系统, 
    其中暂时不要求生成完整的记忆链条, 但是需要对每场对话进行总结.
    并且给出相应的结果, 比如当前人物的状态等. 记忆提取机制同样需要进行改变.
    '''

    def get_user_prompt():
      user_prompt = []
      for id, memory in memory_dict.items():
        format_memory = f"id: {id} 记忆描述: {memory['description']}"
        user_prompt.append(format_memory)

      for id, knowledge in knowledge_dict.items():
        format_knowledge = f"id: {id} 名称: {knowledge['name']} 描述: {knowledge['description']}"
        user_prompt.append(format_knowledge)

      for id, relation in relation_dict.items():
        format_relation = f"{relation['from']}与{relation['to']}: {relation['description']}"
        user_prompt.append(format_relation)

      for chat in chat_content_list:
        user_prompt.append(f"{chat['role']}: {chat['content']}")

      return "\n".join(user_prompt)

    
    def gene_summary_memory_prompt():
      system_prompt = "\n".join([
        "[人物设定]",
        self.get_system_prompt(),
        "[总结任务]",
        summary_memory_prompt,
      ])
      return system_prompt, get_user_prompt()
    
    def gene_summary_knowledge_prompt():   
      system_prompt = "\n".join([
        "[人物设定]",
        self.get_system_prompt(),
        "[总结任务]",
        summary_knowledge_prompt,
      ])
      return system_prompt, get_user_prompt()

    def get_summary_from_model(system_prompt, user_prompt):
      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ]

      response = self.summary_client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=1.3,
        stream=False,
        response_format= {"type": "json_object"}
      )
      return json.loads(response.choices[0].message.content)

    def process_knowledge_summary():
      system_prompt, user_prompt = gene_summary_knowledge_prompt()
      summary = get_summary_from_model(system_prompt, user_prompt)
      print(summary)

      knowledge_items = summary["知识条目"]
      relation_items = summary["关系条目"]

      for item in knowledge_items:
        item["embedding"] = self.memories_encoder([item["名称"]])[0].tolist()
        item["description"] = item["描述"]
        item["name"] = item["名称"]

      knowledge_node_query = '''
        UNWIND $knowledge_items as knowledge
        CREATE (kn: Knowledge {
          name: knowledge.name,
          description: knowledge.description,
          embedding: knowledge.embedding
        })
        RETURN ID(kn) AS knowledge_id
      '''

      with self.driver.session() as session:
        knowledge_ids = session.execute_write(lambda tx: 
          [record["knowledge_id"] 
            for record in tx.run(knowledge_node_query, knowledge_items=knowledge_items).data()]
        )

      for item in relation_items:
        item["description"] = item["关系描述"]
        item["from_name"] = item["知识1"]
        item["to_name"] = item["知识2"]

      relation_edge_query = '''
        UNWIND $relation_items AS relation
        MATCH (from:Knowledge {name: relation.from_name})
        MATCH (to :Knowledge {name: relation.to_name})
        CREATE (from)-[:RELATE_KNOWLEDGE {
          from: relation.from_name,
          to: relation.to_name,
          description: relation.description
        }]->(to)
      '''
      with self.driver.session() as session:
        _ = session.execute_write(lambda tx: 
          tx.run(relation_edge_query, relation_items=relation_items).data()
        )

      for knowledge_id, knowledge_item in zip(knowledge_ids, knowledge_items):
        knowledge_dict[knowledge_id] = {
          "name": knowledge_item["name"],
          "description": knowledge_item["description"],
        }
      

    def process_memory_summary():
      # 在对话结束完毕时, 创建结束全局节点后, 需要对不同的子记忆进行派生
      # 主要会添加部分让角色影响深刻的事情, 以及可能的一些情感波动
      # 需要将相关的知识与记忆推理过程关联起来
      system_prompt, user_prompt = gene_summary_memory_prompt()
      summary = get_summary_from_model(system_prompt, user_prompt)
      print(summary)

      memory_items = summary["记忆条目"]
      summary_items = summary["对话总结"]

      state_dict = summary_items["状态仪表盘"]
      format_state = "\n".join([f"{k}: {v}" for k, v in state_dict.items()])

      task_dict = summary_items["角色任务"]
      format_task = "\n".join([f"{k}: {v}" for k, v in task_dict.items()])

      plan_dict = summary["行动计划"]
      format_plan = "\n".join([f"{k}: {v}" for k, v in plan_dict.items()])

      content_summary = summary_items["内容总结"]

      dialog_node_query = '''
        OPTIONAL MATCH (last:Dialog)
        WHERE NOT (last)-[:NEXT]->()
        WITH last
        CREATE (now:Dialog { 
          timestamp: datetime(), 
          length: COALESCE(last.length, 0) + 1,
          state: $format_state,
          task: $format_task,
          plan: $format_plan,
          content: $content_summary
        })

        WITH last, now
        FOREACH (_ IN CASE WHEN last IS NOT NULL THEN [1] ELSE [] END | 
          CREATE (last)-[:NEXT]->(now)
        )
      '''

      with self.driver.session() as session:
        _ = session.execute_write(lambda tx: 
          tx.run(
            dialog_node_query, 
            format_state=format_state, 
            format_task=format_task, 
            format_plan=format_plan,
            content_summary=content_summary
          ).data()
      )
        
      for item in memory_items:
        item['name'] = item["记忆描述"]
        item["description"] = item["详细记忆"]
        item["embedding"] = self.memories_encoder([item["description"]])[0].tolist()
        
      memory_node_query = '''
        UNWIND $memory_items AS memory
        OPTIONAL MATCH (now:Dialog)
        WHERE NOT (now)-[:NEXT]->()
        WITH now, memory
        CREATE (now)-[:HAS_MEMORY]->(mem:Memory {
          name: memory.name,
          description: memory.description,
          embedding: memory.embedding
        })
        RETURN ID(mem) as memory_id
      '''

      with self.driver.session() as session:
        memory_ids = session.execute_write(lambda tx: 
          [ record["memory_id"] for record in 
            tx.run(memory_node_query, memory_items=memory_items).data()]
        )
      
      knowledge_memory_connect = []
      mem_memory_connect = []

      for memory_id, memory in zip(memory_ids, memory_items):
        for knowledge_id in memory["相关知识"]:
          knowledge_memory_connect.append({"kid": knowledge_id, "mid": memory_id})
        for conc_mem_id in memory["相关记忆"]:
          mem_memory_connect.append({"mid1": conc_mem_id, "mid2": memory_id})
      
      connect_query_know = """
        UNWIND $knowledge_memory_connect AS pair
        MATCH (know:Knowledge) WHERE ID(know) = pair.kid
        MATCH (mem:Memory) WHERE ID(mem) = pair.mid
        CREATE (know)<-[:HAS_KNOWLEDGE]-(mem)
      """

      connect_query_mem = """
        UNWIND $mem_memory_connect AS pair
        MATCH (old:Memory) WHERE ID(old) = pair.mid1
        MATCH (new:Memory) WHERE ID(new) = pair.mid2
        CREATE (old)-[:RELATE_MEMORY]->(new)
      """

      with self.driver.session() as session:
        _ = session.execute_write(lambda tx: 
          tx.run(connect_query_know, knowledge_memory_connect=knowledge_memory_connect)
        )
        _ = session.execute_write(lambda tx: 
          tx.run(connect_query_mem, mem_memory_connect=mem_memory_connect)
        )
    process_knowledge_summary()
    process_memory_summary()
