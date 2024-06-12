from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler

from database import Database
from logger import Logger

SHOW_LOG = True

logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

def read_csv(path: str, spark: SparkSession) -> DataFrame:
    log.info(f'Reading csv: {path}')
    return spark.read.csv(path, header=True, inferSchema=True)

def assemble(df: DataFrame) -> DataFrame:
    log.info(f'Assembling dataframe to vector with schema: {df.schema}')
    vecAssembler = VectorAssembler(inputCols=df.columns, outputCol='features')
    data = vecAssembler.transform(df)

    return data

def scale(df: DataFrame) -> DataFrame:
    log.info(f'Standard scaling dataframe with schema: {df.schema}')
    standardScaler = StandardScaler(inputCol='features', outputCol='scaled')
    model = standardScaler.fit(df)
    data = model.transform(df)

    return data

def drop_columns(df: DataFrame) -> DataFrame:
    log.info(f'Dropping non-numerical columns.')
    cols = [col for col in df.columns if '100g' in col]
    data = df.select(cols)
    
    return data

def dropna(df: DataFrame) -> DataFrame:
    log.info(f'Dropping na values.')
    return df.dropna(thresh=10)

def fillna(df: DataFrame) -> DataFrame:
    log.info(f'Filling na values with zeros.')
    return df.fillna(0.)

def trunc(df: DataFrame) -> DataFrame:
    log.info(f'Trimmimg dataframe down to 1000 samples.')
    return df.limit(1000)

def cast(df: DataFrame) -> DataFrame:
    log.info(f'Casting all columns to double type.')
    return df.select(*(col(c).cast("Double").alias(c) for c in df.columns))

def read_and_prep(path: str, spark: SparkSession) -> DataFrame:
    df = read_csv(path, spark)
    df = drop_columns(df)
    df = dropna(df)
    df = trunc(df)
    df = cast(df)
    df = fillna(df)
    # df = assemble(df)
    # df = scale(df)

    return df

def df_to_dbtable(df: DataFrame, dbtable: str, database: Database):
    database.write_dbtable(df, dbtable)

def dbtable_to_df(dbtable: str, database: Database) -> DataFrame:
    return database.read_dbtable(dbtable)