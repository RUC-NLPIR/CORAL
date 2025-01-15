# <div align="center"><img src="https://github.com/user-attachments/assets/2953f19b-0ec1-4b4a-a911-8956446fb9ab" width="40" />CORAL: Benchmarking Conversational Retrieval-Augmentation Generation<div>


<div align="center">
<a href="https://github.com/Ariya12138/CORAL/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>


We present a large-scale conversational RAG benchmark named **CORAL** and propose a unified framework for standardizing and evaluating various conversational RAG baselines.  

- **CORAL:** CORAL has five critical features: open-domain coverage, knowledge-intensiveness, freeform response generation, handling of topic shifts, and citation labeling.  In CORAL, we evaluate conversational RAG systems across three essential tasks:   
(1) **Conversational Passage Retrieval**: assessing the system’s ability to retrieve relevant information from a large document set based on multi-turn context;  
(2) **Response Generation**: evaluating the system’s capacity to generate accurate, contextually rich answers;  
(3) **Citation Labeling**: ensuring that the generated responses are transparent and grounded by requiring correct attribution of sources.


- **Conversational RAG Framework:** We develop a unified framework for standardizing and evaluating various conversational RAG baselines, facilitating systematic comparison and advancement in this rapidly evolving field.



---

## 🪸 CORAL

### 🌠 Overview of Constructing Dataset Process
<img width="1276" alt="image" src="https://github.com/user-attachments/assets/24e33890-70b9-45de-8a98-469c8b4a97b9">


### 🌈 Four Different Conversation Flow Sampling
<img width="1397" alt="image" src="https://github.com/user-attachments/assets/c65a197e-d097-4509-936a-df1a28118ffd">


### 🎯 Data statistics
<table style="width: 100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th></th>
            <th colspan="2">LDS</th>
            <th colspan="2">SIDS</th>
            <th colspan="2">STRW</th>
            <th colspan="2">DTRW</th>
        </tr>
        <tr>
            <th></th>
            <th>Train</th>
            <th>Test</th>
            <th>Train</th>
            <th>Test</th>
            <th>Train</th>
            <th>Test</th>
            <th>Train</th>
            <th>Test</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td># Conversation</td>
            <td>1800</td>
            <td>200</td>
            <td>1800</td>
            <td>200</td>
            <td>1800</td>
            <td>200</td>
            <td>1800</td>
            <td>200</td>
        </tr>
        <tr>
            <td># Turns</td>
            <td>5934</td>
            <td>651</td>
            <td>16082</td>
            <td>1727</td>
            <td>18165</td>
            <td>1949</td>
            <td>19411</td>
            <td>2153</td>
        </tr>
        <tr>
            <td># Turns / Conversation</td>
            <td>3.30</td>
            <td>3.26</td>
            <td>8.93</td>
            <td>8.64</td>
            <td>10.09</td>
            <td>9.75</td>
            <td>10.78</td>
            <td>10.77</td>
        </tr>
        <tr>
            <td># Tokens / Question</td>
            <td>13.70</td>
            <td>13.89</td>
            <td>12.62</td>
            <td>12.64</td>
            <td>12.72</td>
            <td>12.88</td>
            <td>14.15</td>
            <td>14.75</td>
        </tr>
        <tr>
            <td># Tokens / Response</td>
            <td>233.81</td>
            <td>147.16</td>
            <td>242.54</td>
            <td>155.54</td>
            <td>243.34</td>
            <td>191.60</td>
            <td>300.47</td>
            <td>259.72</td>
        </tr>
        <tr>
            <td># Positive passages/ Turn</td>
            <td>3.25</td>
            <td>2.03</td>
            <td>2.64</td>
            <td>1.73</td>
            <td>3.01</td>
            <td>2.12</td>
            <td>3.98</td>
            <td>3.50</td>
        </tr>
    </tbody>
</table>

### Dataset Format
CORAL includes 8,000 conversations in jsonline format. Each line in either the ```train_conversation.json``` or ```test_conversation.json``` file follows this structure:
```
{

    "conv_id": "Train_type_convid",
    "turns": [
            {
                "turn_id": 1,
                "question": "",
                "response": "",
                "golden_rewrite": "",
                "golden_docs_pids": [],
                "golden_docs_text": []
            },
            {
                "turn_id": 2,
                "question": "",
                "response": "",
                "golden_rewrite": "",
                "golden_docs_pids": [],
                "golden_docs_text": []
            },
    ...
}
```

## 🔥 Conversational RAG Framework


<img width="1274" alt="image" src="https://github.com/user-attachments/assets/e9484f88-695a-40d2-85f4-6a77a25b0b67">

## 🚀 QuickStart


## :bookmark: License

Our code is licensed under the [<u>MIT License</u>](./LICENSE).
Our dataset is distributed under the [CC BY-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.








