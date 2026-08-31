from src.recommender import recommend_detailed


def main():

    print("=" * 60)
    print("       PERSONALIZED ENTERTAINMENT AI")
    print("=" * 60)

    print("\nMovie Recommendation System")
    print("Type 'quit' to exit.\n")

    while True:

        movie = input("Enter a movie: ").strip()

        if movie.lower() == "quit":
            print("\nGoodbye!")
            break

        if not movie:
            print("Please enter a movie name.\n")
            continue

        results = recommend_detailed(movie)

        if not results:
            print(
                f"\nSorry, '{movie}' was not found in the dataset."
            )
            print("Try another movie.\n")
            continue

        print("\nRecommended movies:\n")

        for i, result in enumerate(results, 1):

            print(
                f"{i}. {result['title']}"
            )

            print(
                f"   Score: {result['score']:.4f}"
            )

            print("   Why:")

            for reason in result["reasons"]:
                print(f"      • {reason}")

            print()


if __name__ == "__main__":
    main()