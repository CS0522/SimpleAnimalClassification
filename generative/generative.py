# Author: Chen Shi
# 产生式动物识别

# RULES = [
#     {"if": {"蹄": True, "长尾巴": True, "骑乘或拉车": True}, "then": "马"},
#     {"if": {"羽毛": True, "喙": True, "下蛋": True}, "then": "鸡"},
#     {"if": {"体型小": True, "柔软皮肤": True, "锋利牙齿和爪子": True}, "then": "猫"},
#     {"if": {"体型多样": True, "毛发": True, "锋利牙齿": True, "人类伴侣": True}, "then": "狗"}
# ]
# RULES: list
# RULES[0]: dict
# RULES[0]["if"]: dict

import tkinter as tk

# files and paths
dataset_path = "./dataset"
rules_filename = "rules.data"

'''
Function Class
'''
class GenerativeSystem:
    
    def __init__(self, dataset_path = dataset_path, rules_filename = rules_filename):
        self.__dataset_path = dataset_path
        self.__rules_filename = rules_filename
        # get rules set from file
        self.__rules = self.__read_rules_from_file(self.__dataset_path, self.__rules_filename)
        # get characters dict
        self.__characters = self.__create_characters()
        # get results dict
        self.__results = self.__create_results()
        # combine characters and results
        self.__database = {**self.__characters, **self.__results}
        print("rules: ")
        print(self.__rules)
        print("characters dict: ")
        print(self.__characters)
        print("results dict: ")
        print(self.__results)
        print("database: ")
        print(self.__database)

        # data list
        self.__ifs_list = []
        self.__thens_list = []

    '''
    private: write rules into a file
    '''
    def __write_rules_to_file(self, dataset_path: str, rules_filename: str, rules: list):
        with open(f'{dataset_path}/{rules_filename}', 'w+', encoding = 'utf-8') as rules_file:
            rules_file.write(str(rules))
        print("Writing rules is done. ")

    '''
    private: read rules from a file
    '''
    def __read_rules_from_file(self, dataset_path: str, rules_filename: str):
        with open(f'{dataset_path}/{rules_filename}', 'r', encoding = 'utf-8') as rules_file:
            line = rules_file.read()
            res = eval(line)
        print("Reading rules is done. ")
        return res
    
    '''
    private: create all rules characters
    '''
    def __create_characters(self):
        characters_list = []
        characters_dict = {}
        for rule in self.__rules:
            for key, value in rule["if"].items():
                characters_list.append(key)
        # transfer to dict
        for index, character in enumerate(characters_list):
            characters_dict[index + 1] = character
        return characters_dict
    
    '''
    private: create all rules results
    '''
    def __create_results(self):
        results_list = []
        results_dict = {}
        for rule in self.__rules:
            results_list.append(rule["then"])
        # transfer to dict
        for index, result in enumerate(results_list):
            results_dict[index + len(self.__characters) + 1] = result
        return results_dict
    
    '''
    public: get rules
    '''
    def get_rules(self):
        return self.__rules
    
    '''
    public: get characters
    '''
    def get_characters(self):
        return self.__characters
    
    '''
    public: get characters
    '''
    def get_results(self):
        return self.__results

    '''
    public: insert a rule
    '''
    def insert_rule(self, new_rule: dict):
        # 1. append new_rule (dict type) to rules (list)
        self.rules.append(new_rule)
        # 2. re-write the updated rules into file
        self.__write_rules_to_file(self.__dataset_path, self.__rules_filename)
        # 3. print updated rules
        print("Inserting a new rule is done.")
        print("Current rules: ")
        print(self.__rules)

    '''
    public: delete a rule
    '''
    def delete_rule(self, rule_index: int):
        # 1. delete target rule
        target_delete_rule = self.__rules.pop(rule_index)
        # 2. re-write the updated rules into file
        self.__write_rules_to_file(self.__dataset_path, self.__rules_filename)
        # 3. print the delete rules
        print(f"Deleting rule {rule_index} is done.")
        print("Current rules: ")
        print(self.__rules)

    '''
    public: change rules into different list
    '''
    def create_data_list(self):
        # 存储每个 if
        ifs_list = []
        # 存储每个 then
        thens_list = []
        for rule in self.__rules:
            if_list = []
            for key, value in rule["if"].items():
                if_list.append(key)
            ifs_list.append(if_list)
            thens_list.append(rule["then"])
        # 赋值
        self.__ifs_list = ifs_list
        self.__thens_list = thens_list
    
    '''
    public: reset the process data list
    '''
    def reset_data_list(self):
        self.__ifs_list = []
        self.__thens_list = []

    '''
    public: get process data list
    '''
    def get_data_list(self):
        return self.__ifs_list, self.__thens_list

    '''
    public: 推理机函数
    '''
    def infer_animal(self, features_list, dict_output):
        # 依次进行循环查找并对过程排序
        for index, if_process in enumerate(self.__ifs_list):
            # 判断此过程是否成立
            cnt = 0
            for data in features_list:
                if data in if_process:
                    cnt += 1
            # 过程成立，进入下一步
            if (cnt == len(if_process)):
                # 此过程中结果是否为最终结果，不是将此过程结果加入到过程中
                if self.__thens_list[index] not in self.__results.values():
                    # 弹出过程和此过程结果，因为此过程已经进行过，此结果存入需要查找的过程中
                    result = self.__thens_list.pop(index)
                    process = self.__ifs_list.pop(index)
                    # 结果是否已经存在过程中，存在则重新寻找，不存在则加入过程，并将其存入最终结果
                    if result not in features_list:
                        dict_output['，'.join(process)] = result
                        end_result = self.infer_animal(features_list + [result], dict_output)
                        return end_result
                    # 存在则直接寻找
                    else:
                        end_result = self.infer_animal(features_list, dict_output)
                        return end_result
                # 找到最终结果，取出结果后返回
                else:
                    process = self.__ifs_list.pop(index)
                    dict_output['，'.join(process)] = self.__thens_list[index]
                    return 1
    

'''
UI Class
'''
class MainWindow:

    def __init__(self):
        # main window
        self.root = tk.Tk()

        self.root.title("产生式动物识别系统")

        self.label = tk.Label(self.root, text = "请选择动物特征", bg = "lightyellow", fg = "red", width = 40)
        self.label.grid(row = 0)

        # 创建产生式系统类
        self.gen_sys = GenerativeSystem()

        # 根据 characters 创建 checkboxs
        self.__characters = self.gen_sys.get_characters()
        # animal features list
        self.animal_features = []
        
        # dict: False/True
        self.__checkboxs = {}
        for i in range(len(self.__characters)):
            # 这里相当于是 {0: False, 1: False, 2: False, 3: False, 4: False}
            self.__checkboxs[i] = tk.BooleanVar()
            # 只有被勾选才变为 True
            tk.Checkbutton(self.root, text=self.__characters[i + 1], variable=self.__checkboxs[i]).grid(row = i + 1, sticky = tk.W)

        self.button_start = tk.Button(self.root, text = "开始识别", width = 15, command = self.__do_infer)
        self.button_start.grid(row = len(self.__characters) + 1)

        self.label_for_result = tk.Label(self.root, text = "\n============ 输出结果 ============", relief = tk.FLAT)
        self.label_for_result.grid(row = len(self.__characters) + 2)

        self.text_result = tk.Text(self.root, width = 40, height = 10)
        self.text_result.grid(row = len(self.__characters) + 3)

        self.root.mainloop()
    
    '''
    private: 重置中间过程数据
    '''
    def __reset_process_data(self):
        self.animal_features = []
        self.gen_sys.reset_data_list()

    '''
    private: print to Text
    '''
    def __print_to_text(self, txt_str: str):
        self.text_result.insert(tk.END, txt_str + '\n')

    '''
    private: clear text
    '''
    def __clear_text(self):
        self.text_result.delete("1.0", "end")

    '''
    private: 执行识别特征
    '''
    def __do_infer(self):
        # 更新 data_list
        self.gen_sys.create_data_list()

        # clear text
        self.__clear_text()

        for i in self.__checkboxs:
            # 被勾选：True；未被勾选：False
            if (self.__checkboxs[i].get()):
                self.animal_features.append(self.__characters[i + 1])
        
        dict_output = {}
        result = self.gen_sys.infer_animal(self.animal_features, dict_output)
        
        # 查找成功
        if (result):
            self.__print_to_text('查询成功，推理过程如下：\n')

            for data in dict_output.keys():
                self.__print_to_text(f"{data} -> {dict_output[data]}")
                # final result
                if dict_output[data] in self.gen_sys.get_results().values():

                    self.__print_to_text(f'\n所识别的动物为：{dict_output[data]}')
        # 查找失败
        else:
            self.__print_to_text('条件不足或无匹配规则，查询失败')
        
        # 重置中间过程数据
        self.__reset_process_data()


if __name__ == "__main__": 
    main_window = MainWindow()