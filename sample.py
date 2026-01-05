import spacy

# 1. モデルのロード
nlp = spacy.load('ja_ginza')

# 2. 解析したいテキストを処理
# ここでは単語単体のリストではなく、文脈や単語をnlpオブジェクトに通します
doc = nlp("猫 犬 車")

# 3. 各トークン（単語）を取り出す
token_cat = doc[0]  # 猫
token_dog = doc[1]  # 犬
token_car = doc[2]  # 車

print(f"単語: {token_cat.text}")
print(f"ベクトル次元数: {token_cat.vector.shape}") # 通常は300次元などが表示されます
print("-" * 20)

# 4. 類似度（Cosine Similarity）を計算
# 1.0に近いほど似ている、0に近いほど似ていない
similarity_cat_dog = token_cat.similarity(token_dog)
similarity_cat_car = token_cat.similarity(token_car)

print(f"「{token_cat.text}」と「{token_dog.text}」の類似度: {similarity_cat_dog:.4f}")
print(f"「{token_cat.text}」と「{token_car.text}」の類似度: {similarity_cat_car:.4f}")