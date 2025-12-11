#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 SentenceTransformer 模型缓存
"""

import os
from pathlib import Path
import platform

def check_model_cache():
    """检查本地模型缓存"""
    print("=" * 60)
    print("检查 SentenceTransformer 模型缓存")
    print("=" * 60)
    
    # 确定缓存目录
    if platform.system() == 'Windows':
        cache_base = Path(os.environ.get('USERPROFILE', '')) / '.cache' / 'huggingface'
    else:
        cache_base = Path.home() / '.cache' / 'huggingface'
    
    print(f"\n1. Hugging Face 缓存基础目录: {cache_base}")
    print(f"   目录是否存在: {cache_base.exists()}")
    
    if not cache_base.exists():
        print("\n❌ 缓存目录不存在，模型尚未下载")
        return False
    
    # 检查 hub 目录
    hub_dir = cache_base / 'hub'
    print(f"\n2. Hub 目录: {hub_dir}")
    print(f"   目录是否存在: {hub_dir.exists()}")
    
    if not hub_dir.exists():
        print("\n❌ Hub 目录不存在，模型尚未下载")
        return False
    
    # 查找模型目录（Hugging Face 使用不同的命名规则）
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    # 尝试两种可能的目录名格式
    model_dir_name1 = f'models--sentence-transformers--{model_name.replace("-", "--")}'
    model_dir_name2 = f'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
    
    model_dir = None
    for dir_name in [model_dir_name1, model_dir_name2]:
        potential_dir = hub_dir / dir_name
        if potential_dir.exists():
            model_dir = potential_dir
            break
    
    # 如果还是没找到，搜索包含模型名的目录
    if model_dir is None:
        dirs = list(hub_dir.iterdir())
        matching_dirs = [d for d in dirs if 'paraphrase-multilingual-MiniLM-L12-v2'.replace('-', '') in d.name.replace('-', '').replace('_', '')]
        if matching_dirs:
            model_dir = matching_dirs[0]
    
    print(f"\n3. 模型目录: {model_dir}")
    print(f"   目录是否存在: {model_dir.exists()}")
    
    if not model_dir.exists():
        print("\n❌ 模型目录不存在，模型尚未下载")
        # 列出所有可能的模型目录
        dirs = list(hub_dir.iterdir())
        model_dirs = [d for d in dirs if 'MiniLM' in str(d) or 'paraphrase' in str(d).lower()]
        if model_dirs:
            print(f"\n   找到 {len(model_dirs)} 个相关目录:")
            for d in model_dirs[:5]:
                print(f"     - {d.name}")
        return False
    
    # 检查模型文件
    print(f"\n4. 检查模型文件...")
    
    # 查找 snapshots 目录
    snapshots_dir = model_dir / 'snapshots'
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            latest_snapshot = snapshots[0]
            print(f"   找到快照: {latest_snapshot.name}")
            
            # 检查关键文件
            required_files = [
                'config.json',
                'config_sentence_transformers.json',
                'modules.json',
                '1_Pooling/config.json'
            ]
            
            all_exist = True
            for file_path in required_files:
                full_path = latest_snapshot / file_path
                exists = full_path.exists()
                status = "✓" if exists else "✗"
                print(f"   {status} {file_path}")
                if not exists:
                    all_exist = False
            
            # 检查 pytorch_model.bin 或 model.safetensors
            model_files = list(latest_snapshot.glob('*.bin')) + list(latest_snapshot.glob('*.safetensors'))
            if model_files:
                print(f"   ✓ 找到模型权重文件: {len(model_files)} 个")
                for f in model_files[:3]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"      - {f.name} ({size_mb:.1f} MB)")
            else:
                print(f"   ✗ 未找到模型权重文件")
                all_exist = False
            
            if all_exist:
                print(f"\n✅ 模型缓存完整，可以使用离线模式")
                return True
            else:
                print(f"\n⚠️  模型缓存不完整，部分文件缺失")
                return False
        else:
            print(f"   ✗ 未找到快照目录")
            return False
    else:
        print(f"   ✗ 未找到 snapshots 目录")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("测试模型加载（离线模式）")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        from pathlib import Path
        import platform
        
        # 设置离线模式
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        
        # 方法1: 尝试直接使用本地缓存路径（完全避免网络请求）
        print("\n方法1: 尝试直接使用本地缓存路径...")
        try:
            # 获取缓存目录
            if platform.system() == 'Windows':
                cache_base = Path(os.environ.get('USERPROFILE', '')) / '.cache' / 'huggingface' / 'hub'
            else:
                cache_base = Path.home() / '.cache' / 'huggingface' / 'hub'
            
            model_dir_name = f'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
            model_dir = cache_base / model_dir_name
            
            if model_dir.exists():
                snapshots_dir = model_dir / 'snapshots'
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        local_model_path = snapshots[0]
                        print(f"   找到本地模型路径: {local_model_path}")
                        model = SentenceTransformer(str(local_model_path), device='cpu')
                        print("✅ 方法1成功：直接从本地路径加载，无网络请求")
                        
                        # 测试编码
                        test_text = "这是一个测试"
                        embedding = model.encode([test_text])
                        print(f"✅ 模型编码测试成功，向量维度: {embedding.shape}")
                        return True
        except Exception as e1:
            print(f"   方法1失败: {str(e1)[:100]}")
        
        # 方法2: 使用模型名称 + local_files_only 参数
        print("\n方法2: 尝试使用 local_files_only 参数...")
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', local_files_only=True)
            print("✅ 方法2成功：使用 local_files_only 参数")
            
            # 测试编码
            test_text = "这是一个测试"
            embedding = model.encode([test_text])
            print(f"✅ 模型编码测试成功，向量维度: {embedding.shape}")
            return True
        except TypeError:
            print("   方法2失败: 当前版本不支持 local_files_only 参数")
        except Exception as e2:
            print(f"   方法2失败: {str(e2)[:100]}")
        
        # 方法3: 仅使用环境变量（可能仍会尝试网络请求）
        print("\n方法3: 仅使用环境变量（可能仍会尝试网络请求）...")
        print("   注意：此方法可能会尝试连接网络验证文件")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        print("✅ 方法3成功：使用环境变量（可能有网络请求）")
        
        # 测试编码
        test_text = "这是一个测试"
        embedding = model.encode([test_text])
        print(f"✅ 模型编码测试成功，向量维度: {embedding.shape}")
        return True
        
    except ImportError:
        print("❌ sentence-transformers 未安装")
        return False
    except Exception as e:
        error_msg = str(e)
        if 'not found' in error_msg.lower() or 'cache' in error_msg.lower():
            print(f"❌ 模型缓存不存在或损坏: {error_msg[:200]}")
        else:
            print(f"❌ 模型加载失败: {error_msg[:200]}")
        return False

if __name__ == '__main__':
    cache_exists = check_model_cache()
    print()
    if cache_exists:
        test_model_loading()
    else:
        print("\n建议：运行以下命令下载模型：")
        print("python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\"")

