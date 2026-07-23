# Tool Registry

This file lists scripts available in workspace/tools and how to use them.

## find_skill.py

Find skills relevant to a task using a backend-generated embedding vector.

```
python /workspace/tools/find_skill.py '[0.1, 0.2, 0.3]'
python /workspace/tools/find_skill.py '[0.1, 0.2, 0.3]' --skills-dir /workspace/skills
```

Reads the `name` and `description` frontmatter from every `SKILL.md` under the skills
directory, loads shared `.embed.npy` sidecars, compares them against the supplied
embedding vector, and prints every match that meets the configured similarity threshold.