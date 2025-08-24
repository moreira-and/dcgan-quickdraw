from loguru import logger
from tqdm import tqdm
import time
import typer

from quickdraw import QuickDrawDataGroup

from src.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # -----------------------------------------
    max_drawings: int = 500,
    draw_name: str = "coffee cup",
    # ----------------------------------------------
):
    # -----------------------------------------
    start_time = time.time()
    logger.info(f"Starting dataset processing with max_drawings={max_drawings}...")
    # -----------------------------------------

    qd_group = QuickDrawDataGroup(
        draw_name, recognized=True, max_drawings=max_drawings, refresh_data=False
    )

    print(f"Number of drawings uploaded: {qd_group.drawing_count}")

    save_dir = RAW_DATA_DIR / draw_name.replace(" ", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, drawing in tqdm(enumerate(qd_group.drawings), desc="Saving drawings"):
        img = drawing.image
        img.save(save_dir / f"{draw_name.replace(" ", "_")}_{i}.png")
        if i >= max_drawings:
            break

    # -----------------------------------------
    elapsed_time = time.time() - start_time
    logger.success(f"Processing dataset complete. Elapsed time: {elapsed_time:.2f} seconds")
    # -----------------------------------------


if __name__ == "__main__":
    app()
