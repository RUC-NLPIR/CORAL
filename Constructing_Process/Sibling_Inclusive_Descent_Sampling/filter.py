from bs4 import BeautifulSoup
import json
import re
file_path = ""
write_path = ""


class TreeNode:
    def __init__(self, name,level, passages=None):
        self.name = name
        self.level = level
        self.children = []  # 子节点列表
        self.next_sibling = None  # 指向下一个兄弟节点的引用
        self.passages = passages



def main():
    with open(file_path, "r") as file,open(write_path, "w") as write_file:
        for line in file:
        # 解析每一行的 JSON 数据
            data = json.loads(line)
            html = data["text"]
            title = data["title"]
            root = parse_html_to_tree(html, title)
            if root:
                longest_path = find_longest_path(root)
                if len(longest_path) > 10:
                    write_file.write(json.dumps(data, ensure_ascii=False) + '\n')
    print("done")
            



def parse_html_to_tree(html, title):
    root = TreeNode(title,1)
    spans = [obj for obj in re.finditer("<h2>", html)]
    if len(spans) > 0:
        root_end_idx = spans[0].start()
        root_text = html[:root_end_idx]
        root.passages = root_text
    #print("-----------------")
    
    #print(root.name, root.passages)
    #print("-----------------")
    stack = []  # 初始化一个空栈
    stack.append(root)
    soup = BeautifulSoup(html, 'html.parser')
    
    
    tags = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
    

    
    for i, tag in enumerate(tags):
        level = int(tag.name[1])  # 获取标题的级别，例如<h2>是2
        node = TreeNode(tag.text, level)
        #print(level, node.name)
        
       
        
        # 处理栈中的父子关系
        last_sibling = None
        while stack and stack[-1].level >= level:
            if not last_sibling and stack[-1].level == level:
                last_sibling = stack[-1]
            stack.pop()  # 如果当前标题的级别 <= 栈顶元素，弹出栈顶

        if stack:
            parent_node = stack[-1]  # 当前栈顶是父节点
            parent_node.children.append(node)  # 将当前节点作为子节点加入父节点的children中
            if last_sibling:
                last_sibling.next_sibling = node
            

        stack.append(node)  # 将当前节点和级别入栈
        
       
    
    return stack[0] if stack else None  # 返回根节点



# 寻找最长路径（既有孩子节点又有兄弟节点的情况）
def find_longest_path(node):
    if not node:
        return []

    # 计算沿孩子节点的最长路径
    if node.children:
        longest_child_path = max([find_longest_path(child) for child in node.children], key=len, default=[])
    else:
        longest_child_path = []

    # 计算沿兄弟节点的最长路径
    if node.next_sibling:
        longest_sibling_path = find_longest_path(node.next_sibling)
    else:
        longest_sibling_path = []

    # 比较两种路径，选择较长的路径
    if len(longest_child_path) > len(longest_sibling_path):
        return [node.name] + longest_child_path
    else:
        return [node.name] + longest_sibling_path



if __name__ == '__main__':
    main()