"""
改进的排序模型

解决泛化性能问题：
1. 减少模型容量
2. 增强正则化
3. 改进训练策略
4. 更好的评估指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.stats import spearmanr
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ImprovedRankingModelConfig:
    """改进的排序模型配置"""
    # 模型架构 - 减少容量
    input_dim: int = 64
    hidden_dims: List[int] = (128, 64)  # 减少层数和宽度
    dropout_rate: float = 0.5  # 增加dropout
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # 增加L2正则化
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15  # 增加耐心值
    
    # 损失函数
    margin: float = 1.0
    loss_type: str = "margin_ranking"  # margin_ranking, cross_entropy, listnet
    
    # 预训练
    pretrain_epochs: int = 20
    pretrain_learning_rate: float = 1e-3
    
    # 设备
    device: str = "auto"
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, plateau
    
    # 梯度裁剪
    gradient_clip: float = 1.0

class ImprovedRankingModel(nn.Module):
    """
    改进的排序模型 - 减少容量，增强正则化
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout_rate: float = 0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # 添加批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

class ImprovedMarginRankingLoss(nn.Module):
    """
    改进的Margin Ranking Loss
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # label: 1 if s_i > s_j, 0 if s_i < s_j
        target = label.float() * 2 - 1  # 1 -> 1, 0 -> -1
        loss = torch.relu(self.margin - target * (s_i - s_j))
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class ImprovedCrossEntropyLoss(nn.Module):
    """
    改进的交叉熵损失
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 将pairwise比较转换为二分类
        scores = torch.stack([s_j, s_i], dim=1)  # [batch_size, 2]
        return self.ce_loss(scores, label)

class ImprovedListNetLoss(nn.Module):
    """
    ListNet损失函数
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, s_i: torch.Tensor, s_j: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 计算概率分布
        scores = torch.stack([s_i, s_j], dim=1) / self.temperature
        probs = torch.softmax(scores, dim=1)
        
        # 真实标签分布
        true_probs = torch.zeros_like(probs)
        true_probs[torch.arange(len(label)), label] = 1.0
        
        # KL散度损失
        loss = torch.sum(true_probs * torch.log(true_probs / (probs + 1e-8) + 1e-8), dim=1)
        return torch.mean(loss)

class ImprovedRankingTrainer:
    """改进的排序模型训练器"""
    
    def __init__(self, model: ImprovedRankingModel, config: ImprovedRankingModelConfig):
        self.model = model
        self.config = config
        
        # 设置设备
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 设置学习率调度器
        if config.use_scheduler:
            if config.scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.epochs
                )
            elif config.scheduler_type == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=20, gamma=0.5
                )
            elif config.scheduler_type == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', patience=5, factor=0.5
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # 设置损失函数
        if config.loss_type == "margin_ranking":
            self.criterion = ImprovedMarginRankingLoss(margin=config.margin)
        elif config.loss_type == "cross_entropy":
            self.criterion = ImprovedCrossEntropyLoss()
        elif config.loss_type == "listnet":
            self.criterion = ImprovedListNetLoss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_spearman': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_score = -float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def pretrain_on_weak_labels(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """弱标签预训练"""
        logger.info("开始改进的弱标签预训练...")
        
        # 创建预训练优化器
        pretrain_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.pretrain_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        pretrain_criterion = nn.MSELoss()
        
        for epoch in range(self.config.pretrain_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                features_i = batch['features_i'].to(self.device)
                features_j = batch['features_j'].to(self.device)
                
                # 使用弱标签进行预训练
                # 这里简化处理，实际应该使用真实的弱标签
                scores_i = self.model(features_i)
                scores_j = self.model(features_j)
                
                # 简单的预训练目标：让模型学习区分不同序列
                target = torch.ones_like(scores_i)  # 简化目标
                loss = pretrain_criterion(scores_i, target)
                
                pretrain_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                pretrain_optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features_i = batch['features_i'].to(self.device)
                    features_j = batch['features_j'].to(self.device)
                    
                    scores_i = self.model(features_i)
                    scores_j = self.model(features_j)
                    
                    target = torch.ones_like(scores_i)
                    loss = pretrain_criterion(scores_i, target)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            logger.info(f"预训练 Epoch {epoch + 1}/{self.config.pretrain_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("改进的弱标签预训练完成")
        return {'pretrain_epochs': self.config.pretrain_epochs}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features_i = batch['features_i'].to(self.device)
            features_j = batch['features_j'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            scores_i = self.model(features_i)
            scores_j = self.model(features_j)
            
            # 计算损失
            loss = self.criterion(scores_i, scores_j, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features_i = batch['features_i'].to(self.device)
                features_j = batch['features_j'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                scores_i = self.model(features_i)
                scores_j = self.model(features_j)
                
                # 计算损失
                loss = self.criterion(scores_i, scores_j, labels)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = (scores_i > scores_j).long()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # 收集预测和标签用于Spearman计算
                all_predictions.extend(scores_i.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算Spearman相关系数
        if len(all_predictions) > 1:
            spearman_corr, _ = spearmanr(all_predictions, all_labels)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        else:
            spearman_corr = 0.0
        
        return avg_loss, accuracy, spearman_corr
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """完整的训练流程"""
        logger.info("开始改进的排序模型训练...")
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_accuracy, val_spearman = self.validate_epoch(val_loader)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)
            self.train_history['val_spearman'].append(val_spearman)
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            if self.scheduler is not None:
                if self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_spearman)
                else:
                    self.scheduler.step()
            
            # 早停检查
            current_score = val_spearman  # 使用Spearman作为主要指标
            
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                logger.info(f"保存最佳模型检查点: epoch {epoch}, score {current_score:.4f}")
            else:
                self.patience_counter += 1
            
            # 打印进度
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Spearman: {val_spearman:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型权重")
        
        logger.info("改进的排序模型训练完成")
        return self.train_history
    
    def evaluate_ranking(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估排序模型性能"""
        logger.info("评估改进的排序模型性能...")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                features_i = batch['features_i'].to(self.device)
                features_j = batch['features_j'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                scores_i = self.model(features_i)
                scores_j = self.model(features_j)
                
                # 计算损失
                loss = self.criterion(scores_i, scores_j, labels)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = (scores_i > scores_j).long()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # 收集预测和标签
                all_predictions.extend(scores_i.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader)
        
        # 计算Spearman相关系数
        if len(all_predictions) > 1:
            spearman_corr, _ = spearmanr(all_predictions, all_labels)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        else:
            spearman_corr = 0.0
        
        results = {
            'accuracy': accuracy,
            'spearman': spearman_corr,
            'loss': avg_loss
        }
        
        logger.info("测试结果:")
        logger.info(f"  排序准确率: {accuracy:.4f}")
        logger.info(f"  Spearman相关系数: {spearman_corr:.4f}")
        
        return results
    
    def predict_ranking(self, sequences_features: List[np.ndarray]) -> List[float]:
        """预测排序分数"""
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.tensor(sequences_features, dtype=torch.float32).to(self.device)
            scores = self.model(features_tensor)
            return scores.cpu().numpy().tolist()

def create_improved_ranking_model(config: ImprovedRankingModelConfig) -> Tuple[ImprovedRankingModel, ImprovedRankingTrainer]:
    """创建改进的排序模型和训练器"""
    model = ImprovedRankingModel(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    )
    
    trainer = ImprovedRankingTrainer(model, config)
    
    return model, trainer
