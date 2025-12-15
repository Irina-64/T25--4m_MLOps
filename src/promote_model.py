import mlflow
import mlflow.sklearn
import json
import sys
from typing import Optional, Dict
from datetime import datetime
import os

class ModelPromoter:
    """
    Класс для автоматического продвижения версий моделей по правилам.
    """
    
    def __init__(self, model_name: str, tracking_uri: str = "sqlite:///mlflow.db"):  # ИЗМЕНЕНО
        """
        Инициализация.
        
        Args:
            model_name: Name of the model in registry
            tracking_uri: MLflow tracking server URI
        """
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def get_model_version_metrics(self, version: int) -> Optional[Dict]:
        """Получить метрики для версии модели."""
        try:
            registered_model = self.client.get_registered_model(self.model_name)
            
            for model_version in registered_model.latest_versions:
                if int(model_version.version) == version:
                    # Получить run и его метрики
                    run = self.client.get_run(model_version.run_id)
                    return run.data.metrics
            
            return None
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return None
    
    def apply_promotion_rules(self, version: int, metrics: Dict) -> Optional[str]:
        """
        Применить правила продвижения и вернуть целевой stage.
        
        Правила:
        - roc_auc > 0.85 and accuracy > 0.85 -> Production
        - roc_auc > 0.80 and accuracy > 0.80 -> Staging
        - иначе -> None (Archived)
        
        Args:
            version: Version number
            metrics: Dictionary of metrics
        
        Returns:
            Target stage name or None
        """
        roc_auc = metrics.get('roc_auc', 0)
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        print(f"\n📊 Metrics for version {version}:")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Правила продвижения
        if roc_auc > 0.85 and accuracy > 0.85:
            print(f"\n✅ Promoting to PRODUCTION (roc_auc={roc_auc:.4f}, accuracy={accuracy:.4f})")
            return "Production"
        elif roc_auc > 0.80 and accuracy > 0.80:
            print(f"\n✅ Promoting to STAGING (roc_auc={roc_auc:.4f}, accuracy={accuracy:.4f})")
            return "Staging"
        else:
            print(f"\n⚠️ Not meeting promotion criteria")
            print(f"  Required: roc_auc > 0.80 and accuracy > 0.80")
            return None
    
    def promote_version(self, version: int, target_stage: str) -> bool:
        """
        Переместить версию в указанный stage.
        
        Args:
            version: Version number
            target_stage: Target stage (Staging, Production, Archived)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Получить текущий stage
            registered_model = self.client.get_registered_model(self.model_name)
            current_stage = None
            
            for model_version in registered_model.latest_versions:
                if int(model_version.version) == version:
                    current_stage = model_version.current_stage
                    break
            
            if current_stage == target_stage:
                print(f"✓ Version {version} already in {target_stage}")
                return True
            
            # Если есть другие версии в целевом stage, архивируем их
            if target_stage != "Archived":
                for model_version in registered_model.latest_versions:
                    if (model_version.current_stage == target_stage and 
                        int(model_version.version) != version):
                        print(f"📦 Archiving previous {target_stage} version {model_version.version}")
                        self.client.transition_model_version_stage(
                            name=self.model_name,
                            version=model_version.version,
                            stage="Archived"
                        )
            
            # Переместить версию в новый stage
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=str(version),
                stage=target_stage
            )
            
            print(f"✓ Version {version} moved to {target_stage}")
            return True
            
        except Exception as e:
            print(f"Error promoting version: {e}")
            return False
    
    def auto_promote_latest(self) -> bool:
        """
        Автоматически продвинуть последнюю версию по правилам.
        
        Returns:
            True if promotion successful, False otherwise
        """
        print("="*80)
        print(f"AUTO PROMOTING MODEL: {self.model_name}")
        print("="*80)
        
        try:
            # Попытка получить модель
            try:
                registered_model = self.client.get_registered_model(self.model_name)
            except Exception as e:
                print(f"⚠️ Model '{self.model_name}' not found in registry. Maybe it's not registered yet?")
                print(f"Error: {e}")
                return False
            
            if not registered_model.latest_versions:
                print(f"❌ No versions found for model {self.model_name}")
                return False
            
            # Получить последнюю версию
            latest_version = registered_model.latest_versions[0]
            version = int(latest_version.version)
            
            print(f"\n📦 Latest version: {version}")
            print(f"   Current stage: {latest_version.current_stage}")
            
            # Получить метрики
            metrics = self.get_model_version_metrics(version)
            if not metrics:
                print("❌ Could not get metrics for version")
                return False
            
            # Применить правила
            target_stage = self.apply_promotion_rules(version, metrics)
            
            if target_stage:
                # Переместить версию
                success = self.promote_version(version, target_stage)
                
                if success:
                    # Обновить описание
                    description = f"""
                    Automatically promoted to {target_stage}
                    Timestamp: {datetime.now().isoformat()}
                    ROC-AUC: {metrics.get('roc_auc', 0):.4f}
                    Accuracy: {metrics.get('accuracy', 0):.4f}
                    """
                    
                    self.client.update_model_version(
                        name=self.model_name,
                        version=str(version),
                        description=description.strip()
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error in auto_promote_latest: {e}")
            return False
    
    def promote_specific_version(self, version: int) -> bool:
        """
        Продвинуть конкретную версию по правилам.
        """
        print("="*80)
        print(f"PROMOTING SPECIFIC VERSION: {self.model_name} v{version}")
        print("="*80)
        
        metrics = self.get_model_version_metrics(version)
        if not metrics:
            print(f"❌ Could not get metrics for version {version}")
            return False
        
        target_stage = self.apply_promotion_rules(version, metrics)
        
        if target_stage:
            return self.promote_version(version, target_stage)
        
        return False
    
    def list_all_versions(self):
        """Вывести все версии модели."""
        print("\n" + "="*80)
        print(f"VERSIONS OF MODEL: {self.model_name}")
        print("="*80)
        
        try:
            registered_model = self.client.get_registered_model(self.model_name)
            
            if not registered_model.latest_versions:
                print("No versions found")
                return
            
            print("\n")
            for version in registered_model.latest_versions:
                metrics = self.get_model_version_metrics(int(version.version))
                
                print(f"📦 Version {version.version}")
                print(f"   Stage: {version.current_stage}")
                print(f"   Created: {version.creation_timestamp}")
                if metrics:
                    print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                if version.description:
                    desc = version.description[:100]
                    if len(version.description) > 100:
                        desc += "..."
                    print(f"   Description: {desc}")
                print()
            
        except Exception as e:
            print(f"⚠️ Model '{self.model_name}' not found in registry. Maybe it's not registered yet?")
            print(f"Error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-promote model versions based on metrics")
    parser.add_argument("--model-name", type=str, default="telco_churn_model", help="Model name")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--auto", action="store_true", help="Auto-promote latest version")
    parser.add_argument("--version", type=int, default=None, help="Promote specific version")
    parser.add_argument("--list", action="store_true", help="List all versions")
    
    args = parser.parse_args()
    
    promoter = ModelPromoter(args.model_name, args.tracking_uri)
    
    if args.list:
        promoter.list_all_versions()
    elif args.auto:
        success = promoter.auto_promote_latest()
        sys.exit(0 if success else 1)
    elif args.version:
        success = promoter.promote_specific_version(args.version)
        sys.exit(0 if success else 1)
    else:
        print("Use --auto to auto-promote latest, --version N to promote specific, or --list to list all")
        parser.print_help()


if __name__ == "__main__":
    main()