from core.query_processor import QueryProcessor

def main():
    print("Belgian Bilingual Corporate Agent")
    query = input("Enter your query: ")
    processor = QueryProcessor()
    results = processor.process(query)
    for result in results:
        print(f"\nCompany: {result['company']}")
        print(f"FR Data: {result['fr_data']}")
        print(f"NL Data: {result['nl_data']}")
        print(f"Validation: {result['validation']}")

if __name__ == "__main__":
    main()
