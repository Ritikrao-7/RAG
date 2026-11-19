"""Create the Pinecone index if it does not exist. Safe to run more than once."""

from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    client = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name

    if client.has_index(name):
        print(f"Index {name} already exists")
        return

    client.create_index(
        name=name,
        dimension=settings.embedding_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    print(f"Created index {name}")


if __name__ == "__main__":
    main()
