# Low Memory Text Vector

Same as FastText vectors, load from disk on demand.

This library converts the FastText binary format, or text vector format (from FastText or other implementations) to a SQLite database.
Queries only load part of the matrix, so we don't need the memory to hold the full model.

Not everyone has so much memory only for your NLP models!
