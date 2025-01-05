import networkx as nx
from transformers import pipeline

# 1. 그래프 데이터 생성
def create_graph():
    G = nx.Graph()
    # 노드 추가 (예: 의학 데이터)
    G.add_node("Epithelioid Sarcoma", type="Disease")
    G.add_node("EZH2 gene", type="Gene")
    G.add_node("Tazemetostat", type="Drug")
    # 엣지 추가 (관계)
    G.add_edge("Epithelioid Sarcoma", "Tazemetostat", relation="indication")
    G.add_edge("EZH2 gene", "Tazemetostat", relation="target")
    return G

# 2. 질의 처리
def process_query(graph, query):
    # 간단한 키워드 검색
    results = []
    for node in graph.nodes:
        if query.lower() in node.lower():
            results.append(node)
    return results

# 3. 검색 및 생성기 통합
def generate_answer(graph, query):
    # 검색
    results = process_query(graph, query)
    if not results:
        return "No relevant information found."

    # 노드의 관계 추적
    paths = []
    for result in results:
        for neighbor in graph.neighbors(result):
            paths.append((result, neighbor, graph[result][neighbor]['relation']))
    
    # 생성 (LLM 사용)
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"The query '{query}' relates to the following paths in the graph: {paths}. Explain the relationship."
    answer = generator(prompt, max_length=1024, num_return_sequences=1)[0]['generated_text']
    
    return answer

# 실행
if __name__ == "__main__":
    G = create_graph()
    query = "Epithelioid Sarcoma"
    answer = generate_answer(G, query)
    print("Answer:", answer)
