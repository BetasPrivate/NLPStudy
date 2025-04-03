import openpyxl
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

'''
电影推荐系统
实现基于用户和物品的协同过滤算法
支持数据导入、相似度计算、推荐生成等功能
'''

class MovieRecommender:
    def __init__(self):
        # 初始化推荐系统的数据结构
        self.user_to_rating = {}  # 用户-物品评分矩阵
        self.item_to_name = {}    # 物品ID到名称的映射
        self.similar_users = {}    # 用户相似度矩阵
        self.similar_items = {}    # 物品相似度矩阵
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, rating_path: str, item_path: str, write_excel: bool = False) -> None:
        """加载数据并构建用户-物品评分矩阵"""
        try:
            # 首先加载电影信息
            self._load_item_data(item_path)
            # 然后加载评分数据
            self._load_rating_data(rating_path)
            
            if write_excel:
                self._write_to_excel()
                
            self.logger.info(f"数据加载完成。电影数：{len(self.item_to_name)}，用户数：{len(self.user_to_rating)}")
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise

    def _load_item_data(self, item_path: str) -> None:
        """加载电影数据"""
        with open(item_path, encoding="ISO-8859-1") as f:
            for line in f:
                # 解析电影ID和名称
                item_id, item_name = line.split("|")[:2]
                self.item_to_name[int(item_id)] = item_name.strip()

    def _load_rating_data(self, rating_path: str) -> None:
        """加载用户评分数据"""
        total_movies = len(self.item_to_name)
        with open(rating_path, encoding="ISO-8859-1") as f:
            for line in f:
                # 解析用户评分数据
                user_id, item_id, score, _ = line.strip().split("\t")
                user_id, item_id, score = map(int, (user_id, item_id, score))
                
                # 初始化用户的评分向量
                if user_id not in self.user_to_rating:
                    self.user_to_rating[user_id] = [0] * total_movies
                self.user_to_rating[user_id][item_id - 1] = score

    def _write_to_excel(self) -> None:
        """将评分矩阵写入Excel文件"""
        try:
            wb = openpyxl.Workbook()
            sheet = wb.active
            
            # 写入表头
            header = ["user_id"] + [self.item_to_name[i + 1] for i in range(len(self.item_to_name))]
            sheet.append(header)
            
            # 写入评分数据
            for user_id, ratings in self.user_to_rating.items():
                sheet.append([user_id] + ratings)
            
            wb.save("user_movie_rating.xlsx")
            self.logger.info("评分矩阵已保存到Excel文件")
        except Exception as e:
            self.logger.error(f"Excel写入失败: {str(e)}")

    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        # 避免除零错误
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            return 0
        return np.dot(vector1, vector2) / norm_product

    def compute_similar_users(self, max_users: int = 100) -> None:
        """计算用户之间的相似度"""
        self.similar_users = {}
        score_cache = {}  # 缓存已计算的相似度

        for user_a, ratings_a in self.user_to_rating.items():
            if user_a > max_users:  # 限制计算规模
                continue
                
            similar_users = []
            ratings_a = np.array(ratings_a)
            
            for user_b, ratings_b in self.user_to_rating.items():
                if user_b >= user_a or user_b > max_users:
                    continue

                # 利用缓存避免重复计算
                cache_key = f"{user_a}_{user_b}"
                if cache_key in score_cache:
                    similarity = score_cache[cache_key]
                else:
                    similarity = self.cosine_similarity(ratings_a, np.array(ratings_b))
                    score_cache[cache_key] = similarity
                    score_cache[f"{user_b}_{user_a}"] = similarity

                similar_users.append([user_b, similarity])

            # 按相似度降序排序
            self.similar_users[user_a] = sorted(similar_users, key=lambda x: x[1], reverse=True)

    def compute_similar_items(self) -> None:
        """计算物品之间的相似度"""
        # 构建物品-用户评分矩阵
        item_vectors = defaultdict(lambda: [0] * len(self.user_to_rating))
        
        for user_id, ratings in self.user_to_rating.items():
            for item_id, rating in enumerate(ratings, 1):
                if rating > 0:  # 只考虑有评分的情况
                    item_vectors[item_id][user_id - 1] = rating

        # 计算物品相似度
        self.similar_items = {}
        for item1, vector1 in item_vectors.items():
            similar_items = []
            for item2, vector2 in item_vectors.items():
                if item1 != item2:
                    similarity = self.cosine_similarity(np.array(vector1), np.array(vector2))
                    similar_items.append([item2, similarity])
            self.similar_items[item1] = sorted(similar_items, key=lambda x: x[1], reverse=True)

    def predict_rating(self, user_id: int, item_id: int, method: str = 'item', topn: int = 10) -> float:
        """预测用户对物品的评分"""
        if method == 'user':
            return self._user_based_prediction(user_id, item_id, topn)
        else:
            return self._item_based_prediction(user_id, item_id, topn)

    def _user_based_prediction(self, user_id: int, item_id: int, topn: int) -> float:
        """基于用户的协同过滤预测"""
        if user_id not in self.similar_users:
            return 0

        pred_score = 0
        total_similarity = 0
        
        for similar_user, similarity in self.similar_users[user_id][:topn]:
            rating = self.user_to_rating[similar_user][item_id - 1]
            if rating > 0:  # 只考虑有评分的用户
                pred_score += rating * similarity
                total_similarity += similarity

        return pred_score / (total_similarity + 1e-8)

    def _item_based_prediction(self, user_id: int, item_id: int, topn: int) -> float:
        """基于物品的协同过滤预测"""
        if item_id not in self.similar_items:
            return 0

        pred_score = 0
        total_similarity = 0

        for similar_item, similarity in self.similar_items[item_id][:topn]:
            rating = self.user_to_rating[user_id][similar_item - 1]
            if rating > 0:
                pred_score += rating * similarity
                total_similarity += similarity

        return pred_score / (total_similarity + 1e-8)

    def recommend_movies(self, user_id: int, method: str = 'item', topn: int = 10) -> List[Tuple[str, float]]:
        """为用户推荐电影"""
        if user_id not in self.user_to_rating:
            return []

        # 获取用户未看过的电影
        unseen_items = [item_id + 1 for item_id, rating in enumerate(self.user_to_rating[user_id]) 
                       if rating == 0]

        # 预测评分并排序
        recommendations = []
        for item_id in unseen_items:
            predicted_score = self.predict_rating(user_id, item_id, method, topn)
            if predicted_score > 0:  # 只推荐有把握的电影
                recommendations.append((self.item_to_name[item_id], predicted_score))

        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:topn]

def main():
    """主函数"""
    # 初始化推荐系统
    recommender = MovieRecommender()
    
    try:
        # 加载数据
        recommender.load_data("ml-100k/u.data", "ml-100k/u.item")
        
        # 计算相似度矩阵
        recommender.compute_similar_users()
        recommender.compute_similar_items()
        
        # 交互式推荐
        while True:
            try:
                user_id = int(input("\n请输入用户ID (输入0退出): "))
                if user_id == 0:
                    break
                    
                if user_id not in recommender.user_to_rating:
                    print("用户ID不存在，请重试")
                    continue
                
                # 获取推荐
                recommendations = recommender.recommend_movies(user_id)
                
                # 显示推荐结果
                print("\n为您推荐以下电影：")
                for movie, score in recommendations:
                    print(f"评分: {score:.4f}\t电影: {movie}")
                    
            except ValueError:
                print("输入无效，请输入数字ID")
                
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()