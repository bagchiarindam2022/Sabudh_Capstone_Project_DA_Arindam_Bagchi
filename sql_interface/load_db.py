from database_manager import DatabaseManager

db = DatabaseManager()

db.load_parquet("data/processed_stocks.parquet")

print("\nLATEST PRICES:\n", db.get_latest_prices())

db.close()
