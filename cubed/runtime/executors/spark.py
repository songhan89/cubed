from typing import Any, Optional, Sequence, List, Tuple, Dict, Union
from networkx import MultiDiGraph
from pyspark.sql import SparkSession

from cubed.runtime.pipeline import visit_nodes 
from cubed.runtime.types import Callback, DagExecutor 
from cubed.runtime.utils import handle_operation_start_callbacks, handle_callbacks  
from cubed.spec import Spec 


class SparkExecutor(DagExecutor):
    """An execution engine that uses Apache Spark."""

    @property
    def name(self) -> str:
        return "spark"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs: Any,
    ):

        # Configure Spark memory settings from Spec if provided
        spark_builder = SparkSession.builder
        if spec is not None and hasattr(spec, "allowed_mem") and spec.allowed_mem:
            spark_builder = spark_builder.config("spark.executor.memory", spec.allowed_mem)
            spark_builder = spark_builder.config("spark.driver.memory", spec.allowed_mem)
            spark_builder = spark_builder.config("spark.speculation", "true")
            
        # Create a Spark session
        spark = spark_builder.getOrCreate()

        for name, node in visit_nodes(dag, resume=resume):
            handle_operation_start_callbacks(callbacks, name)
            pipeline = node["pipeline"]
            # Create an RDD from pipeline.mappable.
            rdd = spark.sparkContext.parallelize(pipeline.mappable)
            # Define the transformation; note that this is lazy.
            lazy_rdd = rdd.map(lambda x: pipeline.function(x, config=pipeline.config))
            results = lazy_rdd.collect()  # <-- Trigger computation immediately
            if callbacks is not None:
                for result in results:
                    handle_callbacks(callbacks, result, {"name": name})
