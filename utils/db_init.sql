-- PostgreSQL + pgvector 数据库初始化脚本
-- 用于存储论文元数据和向量嵌入

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建论文主表（结构化元数据）
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    paper_id TEXT UNIQUE NOT NULL,  -- 唯一标识符（基于文件路径的哈希）
    title TEXT,
    authors TEXT[],  -- 作者数组
    abstract TEXT,
    year INTEGER,
    journal TEXT,
    keywords TEXT[],
    doi TEXT,
    url TEXT,
    source TEXT,  -- 'csv', 'email', 'pdf', 'zotero', 'obsidian'
    source_id TEXT,  -- 原始数据源ID
    attachment_path TEXT,  -- PDF文件路径
    zotero_key TEXT,  -- Zotero条目key
    obsidian_note_path TEXT,  -- Obsidian笔记路径
    metadata JSONB,  -- 灵活存储额外信息
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 创建向量块表（用于RAG）
CREATE TABLE IF NOT EXISTS paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384),  -- pgvector类型，384维（paraphrase-multilingual-MiniLM-L12-v2）
    chunk_size INTEGER,
    metadata JSONB,  -- 额外的块元数据
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(paper_id, chunk_index)  -- 确保每个论文的块索引唯一
);

-- 创建索引
-- 论文表索引
CREATE INDEX IF NOT EXISTS idx_papers_paper_id ON papers(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_created_at ON papers(created_at);

-- 向量块表索引
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON paper_chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON paper_chunks(paper_id, chunk_index);

-- 向量相似度搜索索引（使用IVFFlat算法，适合大规模数据）
-- 注意：需要先有数据才能创建此索引
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks 
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建触发器（如果不存在）
-- 注意：PostgreSQL不支持CREATE TRIGGER IF NOT EXISTS，需要在Python代码中检查
-- 这里先删除可能存在的触发器，然后重新创建
DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 创建视图：论文统计
CREATE OR REPLACE VIEW paper_stats AS
SELECT 
    source,
    COUNT(*) as paper_count,
    COUNT(DISTINCT paper_id) as unique_papers,
    MIN(created_at) as first_added,
    MAX(created_at) as last_added
FROM papers
GROUP BY source;

-- 创建视图：论文详细信息（包含块数量）
CREATE OR REPLACE VIEW paper_details AS
SELECT 
    p.*,
    COUNT(pc.id) as chunk_count,
    SUM(LENGTH(pc.chunk_text)) as total_text_length
FROM papers p
LEFT JOIN paper_chunks pc ON p.paper_id = pc.paper_id
GROUP BY p.id;

-- 创建对话历史表（用于存储用户与助手的对话记录）
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT,  -- 会话ID（可选，用于区分不同会话）
    user_message TEXT NOT NULL,  -- 用户消息
    assistant_message TEXT NOT NULL,  -- 助手回复
    created_at TIMESTAMP DEFAULT NOW(),  -- 创建时间
    metadata JSONB  -- 额外元数据（如使用的工具、查询的论文等）
);

-- 创建对话历史索引
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);

