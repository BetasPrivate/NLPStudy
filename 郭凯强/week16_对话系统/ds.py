import json                  # 导入json模块，用于处理JSON格式的场景配置文件
import pandas               # 导入pandas模块，用于读取Excel格式的模板文件
import re                   # 导入re模块，用于正则表达式匹配


'''
对话系统
基于场景脚本完成多轮对话
'''

class DialogSystem:
    def __init__(self):
        self.load()         # 初始化时加载所有必要的配置和数据

    def load(self):
        self.all_node_info = {}  #key = 节点id， value = node info
        self.load_scenrio("scenario-买衣服.json")      # 加载买衣服场景的对话节点
        self.load_scenrio("scenario-看电影.json")      # 加载看电影场景的对话节点
        self.slot_info = {} #key = slot, value = [反问，可能取值]
        self.load_templet()                           # 加载槽位填充模板

        #初始化一个专门的节点用于实现在任意时刻的重听
        self.init_repeat_node()                       # 初始化重听功能节点
    
    #实现思路：一个重听节点可以是所有节点的子节点
    def init_repeat_node(self):
        node_id = "special_repeat_node"               # 定义重听节点的唯一标识符
        node_info = {"id":node_id, "intent":["你说啥", "再说一遍"]}  # 定义重听节点的信息，包含触发意图
        self.all_node_info[node_id] = node_info      # 将重听节点添加到节点信息字典中
        for node_info in self.all_node_info.values(): # 遍历所有现有节点
            node_info["childnode"] = node_info.get("childnode", []) + [node_id]  # 将重听节点添加为每个节点的子节点
    
    def init_memory(self):
        memory = {}                                   # 初始化对话记忆字典
        memory["available_node"] = ["scenario-买衣服-node1", "scenario-看电影-node1"]  # 设置初始可用节点为两个场景的入口节点
        return memory
    
    def load_scenrio(self, path): 
        scenario_name = path.replace(".json", "")     # 获取场景名称（去除.json后缀）
        with open(path, "r", encoding="utf-8") as f:  # 打开并读取场景配置文件
            scenario_data = json.load(f)              # 解析JSON数据
        for node_info in scenario_data:               # 遍历场景中的每个节点
            node_id = node_info["id"]                 # 获取节点ID
            node_id = scenario_name + "-" + node_id   # 构造完整的节点ID（场景名-节点ID）
            if "childnode" in node_info:              # 如果节点有子节点
                node_info["childnode"] = [scenario_name + "-" + child for child in node_info["childnode"]]  # 更新子节点ID
            self.all_node_info[node_id] = node_info   # 将节点信息存入字典
    
    def load_templet(self):
        df = pandas.read_excel("./slot_fitting_templet.xlsx")  # 读取Excel格式的槽位模板文件
        for i in range(len(df)):                      # 遍历每一行
            slot = df["slot"][i]                      # 获取槽位名称
            query = df["query"][i]                    # 获取询问语句
            values = df["values"][i]                  # 获取可能的取值模式
            self.slot_info[slot] = [query, values]    # 存储槽位信息

    def run(self, query, memory):
        if memory == {}:                              # 如果是新对话
            memory = self.init_memory()               # 初始化对话记忆
        memory["query"] = query                       # 记录用户输入
        memory = self.nlu(memory)                     # 自然语言理解
        memory = self.dst(memory)                     # 对话状态追踪
        memory = self.pm(memory)                      # 对话策略管理
        memory = self.nlg(memory)                     # 自然语言生成
        return memory
    
    def nlu(self, memory):
        # 语义解析
        memory = self.get_intent(memory)              # 获取用户意图
        memory = self.get_slot(memory)                # 提取槽位值
        return memory
    
    def get_intent(self, memory):
        # 获取意图
        hit_node = None                               # 初始化命中的节点
        hit_score = -1                                # 初始化最高得分
        for node_id in memory["available_node"]:      # 遍历所有可用节点
            score = self.get_node_score(node_id, memory)  # 计算节点得分
            if score > hit_score:                     # 如果得分更高
                hit_node = node_id                    # 更新命中节点
                hit_score = score                     # 更新最高得分
        memory["hit_node"] = hit_node                # 记录命中的节点
        memory["hit_score"] = hit_score              # 记录命中的得分
        return memory
    
    def get_node_score(self, node_id, memory):
        #计算意图得分
        intent_list = self.all_node_info[node_id]["intent"]  # 获取节点的意图列表
        query = memory["query"]                       # 获取用户输入
        scores = []                                   # 存储所有意图的得分
        for intent in intent_list:                    # 遍历所有意图
            score = self.similarity(query, intent)    # 计算相似度得分
            scores.append(score)                      # 添加到得分列表
        return max(scores)                            # 返回最高得分
    
    def similarity(self, query, intent):
        #文本相似度计算，使用jaccard距离
        intersect = len(set(query) & set(intent))     # 计算交集大小
        union = len(set(query) | set(intent))         # 计算并集大小
        return intersect / union                      # 返回Jaccard相似度

    def get_slot(self, memory):
        # 获取槽位
        hit_node = memory["hit_node"]                 # 获取命中的节点
        for slot in self.all_node_info[hit_node].get("slot", []):  # 遍历节点需要的槽位
            if slot not in memory:                    # 如果槽位未填充
                values = self.slot_info[slot][1]      # 获取槽位的可能取值模式
                info = re.search(values, memory["query"])  # 在用户输入中搜索匹配
                if info is not None:                  # 如果找到匹配
                    memory[slot] = info.group()       # 记录槽位值
        return memory

    def dst(self, memory):
        # 对话状态跟踪
        hit_node = memory["hit_node"]                 # 获取命中的节点
        for slot in self.all_node_info[hit_node].get("slot", []):  # 检查所需槽位
            if slot not in memory:                    # 如果有未填充的槽位
                memory["require_slot"] = slot         # 记录需要询问的槽位
                return memory
        memory["require_slot"] = None                 # 所有槽位都已填充

        if hit_node == "special_repeat_node":         # 特殊处理重听
            memory["state"] = "repeat"                # 设置重听状态
        else:
            memory["state"] = None                    # 清除状态
        return memory

    def pm(self, memory):
        # 对话策略执行
        if memory["require_slot"] is not None:        # 如果需要填充槽位
            #反问策略
            memory["available_node"] = [memory["hit_node"]]  # 保持在当前节点
            memory["policy"] = "ask"                  # 设置反问策略
        elif memory["state"] == "repeat":             # 如果是重听请求
            #重听策略  不对memory做修改，只更新policy
            memory["policy"] = "repeat"               # 设置重听策略
        else:
            #回答
            # self.system_action(memory)              # 系统动作完成下单，查找等
            memory["available_node"] = self.all_node_info[memory["hit_node"]].get("childnode", [])  # 更新可用节点为当前节点的子节点
            memory["policy"] = "answer"               # 设置回答策略
        return memory

    def nlg(self, memory):
        # 自然语言生成  
        if memory["policy"] == "ask":                 # 如果是反问策略
            slot = memory["require_slot"]             # 获取需要询问的槽位
            reply = self.slot_info[slot][0]           # 获取反问模板
        elif memory["policy"] == "repeat":            # 如果是重听策略
            #使用上一轮的回复
            reply = memory["reply"]                   # 重复上一次的回复
        else:
            reply = self.all_node_info[memory["hit_node"]]["response"]  # 获取节点的回复模板
            reply = self.replace_templet(reply, memory)  # 替换模板中的槽位
        memory["reply"] = reply                       # 记录回复内容
        return memory

    def replace_templet(self, reply, memory):
        #替换模板中的槽位
        hit_node = memory["hit_node"]                 # 获取命中的节点
        for slot in self.all_node_info[hit_node].get("slot", []):  # 遍历节点的槽位
            reply = re.sub(slot, memory[slot], reply)  # 用实际值替换槽位标记
        return reply

if __name__ == '__main__':
    ds = DialogSystem()                               # 创建对话系统实例
    memory = {}                                       # 初始化对话记忆
    while True:                                       # 持续对话循环
        query = input("用户输入：")                    # 获取用户输入
        memory = ds.run(query, memory)                # 运行对话系统
        print(memory["reply"])                        # 输出系统回复
        print()                                       # 打印空行