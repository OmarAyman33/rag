from langchain_text_splitters import RecursiveCharacterTextSplitter

document = """
Question (2): [14 marks]
a) Imagine a boss battle in an Role Playing Game where the player (Player 1) faces an AI boss (Player 2). The battle has two stages:

Stage 1: The player chooses to either "Attack" or "Defend."
Stage 2: Depending on the player's choice, the boss reacts by either "Counter-Attack" or "Heal."
If the player chooses "Attack" and the boss chooses "Counter-Attack," the player loses 50 HP, and the boss loses 30 HP. If the player chooses "Attack" and the boss chooses "Heal," the player loses 20 HP, and the boss gains 20 HP. If the player chooses "Defend" and the boss chooses "Counter-Attack," the player loses 10 HP, and the boss loses 10 HP. If the player chooses "Defend" and the boss chooses "Heal," the player loses 5 HP, and the boss gains 10 HP.

Draw the extensive form representation of the game. [2 Marks]
Use backward induction to find the Sub-game Perfect Nash Equilibrium of the game. [2 Marks]
Extensive form game tree for a boss battle. The root node is for the player, who can choose 'Attack' or 'Defend'. Choosing 'Attack' leads to a node for the boss, who can choose 'counter' (resulting in payoffs (-50, -30)) or 'heal' (resulting in payoffs (-20, 20)). Choosing 'Defend' leads to another node for the boss, who can choose 'counter' (resulting in payoffs (-10, -10)) or 'heal' (resulting in payoffs (-5, 10)). Backward induction arrows point to the optimal choices: 'heal' for the boss in the 'Attack' branch, 'counter' for the boss in the 'Defend' branch, and 'Defend' for the player.
graph TD
    Player((Player)) -- Attack --> Boss1((Boss))
    Player -- Defend --> Boss2((Boss))
    Boss1 -- counter --> P1["(-50, -30)"]
    Boss1 -- heal --> P2["(-20, 20)"]
    Boss2 -- counter --> P3["(-10, -10)"]
    Boss2 -- heal --> P4["(-5, 10)"]
  
(assuming each side is attempting to maximize their hp)

Nash Equilibrium:

player defends, boss heals
"""


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
texts = text_splitter.split_text(document)

for num, chunk in enumerate(texts, start=1):
    print(f"chunk {num}: {chunk}\n")


from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])
