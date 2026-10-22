# Data Notes

The final RAG corpus was generated from legally obtained local HTML service-manual folders for:

- 2001 Toyota Tundra Limited, 4WD
- 2001 Lexus LS 430

The public repository does not redistribute the full manual text or the derived 40 MB chunk CSV. Run `notebooks/Cela_Claudio_FINAL.ipynb` with the local manual folders to create:

```text
toyota_lexus_rag_chunks.csv
toyota_lexus_embeddings.npy
```

The generated chunk file contains 15,418 service-manual rows and includes fields such as vehicle make, model, year, trim, source type, file path, page title, breadcrumb path, section type, warnings, notes, image-caption metadata, chunk index, chunk ID, and chunk text.

A small forum-style dataset was tested separately but is not part of the final system because the service-manual-only implementation produced stronger final benchmark scores.