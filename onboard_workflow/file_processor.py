import json
import asyncio
from config.config import mistralai_config
import logging


class FileProcessor:
    def __init__(self, files, max_concurrent=3):
        self.files = files
        self.semaphore = asyncio.Semaphore(max_concurrent)  # throttle concurrency
        self.app = mistralai_config

    async def _perform_ocr(self, url: str):
        """
        Performs OCR for PDF and image files
        """
        logging.info(f"Performing OCR on file: {url}")
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = self.app.ocr.process(
                        model="mistral-ocr-latest",
                        document={"type": "document_url", "document_url": url}
                    )
                    return json.loads(response.model_dump_json())

                except Exception as e1:
                    try:
                        response = self.app.ocr.process(
                            model="mistral-ocr-latest",
                            document={"type": "image_url", "image_url": url}
                        )
                        return json.loads(response.model_dump_json())

                    except Exception as e2:
                        if "429" in str(e2) and attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return str(e2)

    async def process_files(self):
        """
        Async task to help process multiple files at once without exceeding rate limits
        """
        tasks = [self._perform_ocr(url) for url in self.files]
        return await asyncio.gather(*tasks)