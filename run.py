from src.pipeline_tdm import TDM_Pipeline_Run
# from src.star_schema import TDM_Pipeline_Run
from src.logging import logger

if __name__ == '__main__':
    try:
        logger.info(f"x=====x>>>>>> Welcome to Synthetic Data Generator <<<<<<x=====x\n")
        object = TDM_Pipeline_Run()
        object.main()
        logger.info(f"x=====x>>>>>> All Stages Execute Successfully <<<<<<x=====x\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
