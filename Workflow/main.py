from config import RESEARCH_REQUEST, configure_dspy
from pipeline import run_pipeline


def main() -> None:
    configure_dspy()
    run_pipeline(RESEARCH_REQUEST)


if __name__ == "__main__":
    main()
