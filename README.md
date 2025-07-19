   # Check the shape of the DataFrame
   print("Shape:", df.shape)

   # Check the data types of each column
   print("\nData Types:\n", df.dtypes)

   # Check for missing values
   print("\nMissing Values:\n", df.isnull().sum())  # or df.isna().sum()

   # Get summary statistics for numerical columns
   print("\nSummary Statistics:\n", df.describe())