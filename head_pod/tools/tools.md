# Tool Registry

This file lists scripts available in workspace/tools and how to use them.

## find_skill.py

Find skills relevant to a task using semantic search.

```
python /workspace/tools/find_skill.py "your query here"
python /workspace/tools/find_skill.py "your query here" --top 3
python /workspace/tools/find_skill.py "your query here" --skills-dir /workspace/skills
```

Reads the `name` and `description` frontmatter from every `SKILL.md` under the skills
directory, generates multilingual embeddings (cached as `.embed.npy` beside each
`SKILL.md`), and prints the top N matches with name, description, and path.