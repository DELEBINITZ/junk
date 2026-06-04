-- Conversation memory: per-session summarization watermark.
--
-- ``summarized_upto`` records how many of the OLDEST messages in a session are
-- already folded into ``summary``. The rolling summarizer compacts only the turns
-- that have been EVICTED from the live history window and sit above this mark, so a
-- long chat loses nothing (old turns enter the summary before they leave the window)
-- and nothing is re-summarized. Backfills to 0 for existing rows (treated as
-- "nothing summarized yet"), which is safe — the next summarize pass catches up.
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS summarized_upto INTEGER NOT NULL DEFAULT 0;
