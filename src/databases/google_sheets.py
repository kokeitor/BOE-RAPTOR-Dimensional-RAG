import gspread
import pandas as pd
import logging
import time
import random
from typing import Union, List
from ETL.models import ClassifyChunk
from databases.utils import get_current_spanish_date_iso

# Logger initializer
logger = logging.getLogger(__name__)

class GoogleSheet:
    """
    Google sheet used as a BBDD schema:
    - BBDD fields or columns must start at A1 cell
    - Always include a last field 'tst_insertion' inside the sheet, that is automatically fill inside this class at
        the time of insert a record
    """
    COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    def __init__(self, credentials: Union[str, dict], document: str, sheet_name: str, api_call_max_tries : int = 15 ):
        self.gc = gspread.service_account(credentials)  # from a json file
        # self.gc = gspread.service_account_from_dict(credentials)  # from a python dict
        self.sh = self.gc.open(document)
        self.sheet = self.sh.worksheet(sheet_name)
        self.api_call_max_tries = api_call_max_tries

    @staticmethod
    def get_record(chunk: ClassifyChunk) -> list:
        return [
            chunk.text,
            chunk.file_path,
            chunk.file_name,
            chunk.file_type,
            chunk.file_size,
            chunk.creation_date,
            chunk.last_modified_date,
            chunk.fecha_publicacion_boe,
            ' # '.join([orden for orden in chunk.orden]),
            ' # '.join([real_decreto for real_decreto in chunk.real_decreto]),
            ' # '.join([ministerios for ministerios in chunk.ministerios]),
            chunk.pdf_id,
            chunk.chunk_id,
            chunk.num_tokens,
            chunk.num_caracteres,
            ' # '.join([f"{label}:{label_id}" for label_id,label in chunk.label2id.items()]),
            ' # '.join([label_id for label_id in chunk.label])
        ]

    def exponential_backoff(self, func, *args, **kwargs):
        retries = 0
        while retries < self.api_call_max_tries:
            try:
                return func(*args, **kwargs)
            except gspread.exceptions.APIError as e:
                if "Quota exceeded" in str(e):
                    retries += 1
                    wait_time = min((2 ** retries) + random.uniform(0, 1), 64)
                    logger.warning(f"Quota exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        raise Exception("Max retries exceeded")

    def read_data(self, range):  # range = "A1:E1". Data devolverá un array de la fila 1 desde la columna A hasta la E
        return self.exponential_backoff(self.sheet.get, range)

    def read_data_by_uid(self, uid):
        data = self.exponential_backoff(self.sheet.get_all_records)
        df = pd.DataFrame(data)
        print(df)
        filtered_data = df[df['id'] == uid]
        return filtered_data  # devuelve un data frame de una tabla, de dos filas siendo la primera las cabeceras de las columnas y la segunda los valores filtrados para acceder a un valor en concreto df["nombre"].to_string()

    def write_data(self, range: str, values: List[List]):  # range ej "A1:V1". values must be a list of list
        tst_insertion = get_current_spanish_date_iso()
        total_fields = self.get_total_fields()
        validated_values = self.validate_records(values=values)
        inserted_values = []
        for record in validated_values:
            for field_index, _ in enumerate(record):
                if field_index + 1 == total_fields - 1:
                    record.append(tst_insertion)
            inserted_values.append(record)
        self.exponential_backoff(self.sheet.update, range, inserted_values)

    def write_data_by_uid(self, uid, values):
        # Find the row index based on the UID
        cell = self.exponential_backoff(self.sheet.find, uid)
        row_index = cell.row
        logger.info(f"write_data_by_uid row to update: {row_index}")
        logger.info(f"write_data_by_uid cell to update: {cell}")

        # Update the row with the specified values
        first_column = GoogleSheet.COLUMNS[0]
        last_column = GoogleSheet.COLUMNS[self.get_total_fields() - 1]
        range_to_update = f"{first_column}{row_index}:{last_column}{row_index}"
        logger.info(f"write_data_by_uid -> range_to_update: {range_to_update}")

        self.exponential_backoff(self.sheet.update, range_to_update, values)

    def get_last_row_range(self):
        last_row = len(self.exponential_backoff(self.sheet.get_all_values)) + 1
        data = self.exponential_backoff(self.sheet.get_all_values)
        range_start = f"A{last_row}"
        range_end = f"{chr(ord('A') + len(data[0]) - 1)}{last_row}"
        return f"{range_start}:{range_end}"

    def get_all_values(self):
        # self.sheet.get_all_values() # this return a list of list, so the get all records is easier to get values filtering
        return self.exponential_backoff(self.sheet.get_all_records)  # this returns a list of dictionaries so the key is the name column and the value is the value for that particular column

    def get_total_fields(self) -> int:
        return len(self.exponential_backoff(self.sheet.get_all_values)[0])

    def get_field_names(self) -> List[str]:
        return self.exponential_backoff(self.sheet.get_all_values)[0]

    def get_total_records(self) -> int:
        return len(self.exponential_backoff(self.sheet.get_all_values))

    def validate_records(self, values: List[List]) -> List[List]:
        """Add "" as field values to complete the number of total BBDD fields"""
        total_fields = self.get_total_fields()
        if isinstance(values, list):
            for record in values:
                if isinstance(record, list):
                    if len(record) < total_fields:
                        extension_length = total_fields - len(record) - 1  # deja hueco para el tsts_insertion
                        record.extend([""] * extension_length)
                    elif len(record) > total_fields:
                        logger.warning("Warning: The current record fields are greater than the actual BBDD fields.")
                else:
                    logger.error(f"Record not a list type -> {record}")
            return values
        else:
            logger.error(f"Trying inserting not a list[list] type -> {values}")

