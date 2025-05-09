import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 配置文件路径
CONFIG = {
    'feature_files': {
        'feature1': '../data/chemical_substructure_vector_processed.csv',
        'feature2': '../data/indication_vector_processed.csv',
        'feature3': '../data/target_vector_processed.csv'
    },
    'pos_samples': 'twosides_interactions.csv',
    'neg_samples': 'reliable_negatives.csv',
    'output_dir': '../dataset_double_tower_model/'
}


def load_features():
    """加载所有特征数据到字典"""
    feature_data = {}
    for name, path in CONFIG['feature_files'].items():
        df = pd.read_csv(path, index_col=0)
        feature_data[name] = df.apply(lambda x: x.values.tolist(), axis=1).to_dict()
    return feature_data


def get_drug_vector(drug, features):
    """获取单个药物的完整特征向量"""
    try:
        return features['feature1'][drug] + features['feature2'][drug] + features['feature3'][drug]
    except KeyError as e:
        print(f"警告：药物 {drug} 缺少特征数据: {e}")
        return None


def build_dataset(sample_file, label, features):
    """构建带标签的数据集"""
    samples = []
    df = pd.read_csv(sample_file, header=None)

    for _, row in df.iterrows():
        drugA, drugB = row[0], row[1]
        vecA = get_drug_vector(drugA, features)
        vecB = get_drug_vector(drugB, features)

        if vecA and vecB:
            samples.append({
                'drugA_features': vecA,
                'drugB_features': vecB,
                'label': label
            })

    return samples


def main():
    # 加载特征数据
    features = load_features()

    # 构建正负样本
    pos_data = build_dataset(CONFIG['pos_samples'], 1, features)
    neg_data = build_dataset(CONFIG['neg_samples'], 0, features)
    all_data = pos_data + neg_data
    df = pd.DataFrame(all_data)

    # 分割数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 保存为Numpy格式
    np.savez(CONFIG['output_dir'] + 'train_data.npz',
             drugA=train_df['drugA_features'].tolist(),
             drugB=train_df['drugB_features'].tolist(),
             labels=train_df['label'].values)

    np.savez(CONFIG['output_dir'] + 'test_data.npz',
             drugA=test_df['drugA_features'].tolist(),
             drugB=test_df['drugB_features'].tolist(),
             labels=test_df['label'].values)

    print(f"数据集构建完成，总样本数: {len(df)}")
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")


if __name__ == "__main__":
    main()