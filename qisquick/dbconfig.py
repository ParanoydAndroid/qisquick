import logging
import pickle
import sqlite3
import traceback

from typing import Dict, List, Tuple

from qls.circuits import TestCircuit
from qls.statblock import Statblock
from qls.qis_logger import get_module_logger

logger = get_module_logger(__name__)

db_location = 'data/circuit_data.sqlite'


def set_db_location(location: str) -> None:
    global db_location
    if location is not None:
        db_location = location


def insert_in_progress(db: str, jobs: List[TestCircuit]) -> None:
    running_records = []
    for tc in jobs:
        uuid = tc.id
        job_id = tc.stats.job_id
        name = tc.stats.name
        serialized = pickle.dumps(tc)

        running_records.append((uuid, job_id, name, serialized))

    command = '''INSERT INTO Running(uuid, jobID, name, object)
                    VALUES(?, ?, ?, ?)'''
    try:
        with sqlite3.connect(db) as conn:
            conn.executemany(command, running_records)
    except sqlite3.DatabaseError:
        logger.error(f'Database save error in table Running: {traceback.format_exc()}')
    finally:
        if conn: conn.close()


def load_in_progress(db: str) -> Dict[str, TestCircuit]:
    check = '''SELECT uuid, jobID, object
                FROM Running
            '''

    jobs = dict()

    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.execute(check)
            records = c.fetchall()
            c.close()
            for record in records:
                jobs[record['uuid']] = pickle.loads(record['object'])
    except sqlite3.DatabaseError:
        logger.error(f'Database load error in table Running: {traceback.format_exc()}')
    finally:
        if conn: conn.close()

    return jobs


def drop_in_progress(db: str, done: List[str]) -> None:
    command = '''DELETE FROM Running
                    WHERE uuid=?'''
    try:
        with sqlite3.connect(db) as conn:
            conformed_done = [(uuid,) for uuid in done]
            conn.executemany(command, conformed_done)
    except sqlite3.DatabaseError:
        msg = 'Error deleting completed jobs'
        error = traceback.format_exc()
        logger.error(f'{msg}: {error}')
    finally:
        if conn: conn.close()


def write_objects(db: str, tcs: List[TestCircuit]) -> None:
    insertions, updates = partition_writes(db, tcs)

    if updates:
        update_objects(db, updates)
        msg = 'Updated record for object(s):\n'
        for tc in updates:
            msg += f'\t{tc.id}\n'

        logger.info(msg)

    if insertions:
        insert_objects(db, insertions)
        msg = 'Inserted record for object(s):\n'
        for tc in insertions:
            msg += f'\t{tc.id}\n'

        logger.info(msg)


def insert_objects(db: str, tcs: List[TestCircuit]) -> None:
    records = []
    for tc in tcs:
        uuid = tc.stats.id
        job_id = tc.stats.job_id
        name = tc.stats.name
        serialized = pickle.dumps(tc)
        notes = tc.stats.notes

        records.append((uuid, job_id, name, serialized, notes))

    command = '''INSERT INTO Circs(uuid, jobID, name, object, notes)
                    VALUES(?, ?, ?, ?, ?)'''

    try:
        with sqlite3.connect(db) as conn:
            conn.executemany(command, records)
    except sqlite3.DatabaseError:
        logger.error(f'Database save error in table Circs: {traceback.format_exc()}')
    finally:
        if conn: conn.close()


def partition_writes(db: str, tcs: List[TestCircuit]) -> Tuple[List, List]:
    check = '''SELECT uuid
                FROM CIRCS
                WHERE uuid=?'''

    updates = []
    insertions = []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            for tc in tcs:
                if conn.execute(check, (tc.id,)).fetchone() is None:
                    insertions.append(tc)
                else:
                    updates.append(tc)
    except sqlite3.DatabaseError:
        logger.error(f'Database retrieval error on Table Circs: {traceback.format_exc()}')
    finally:
        if conn: conn.close()

    return insertions, updates


def update_objects(db: str, tcs: List[TestCircuit]) -> None:
    records = []
    for tc in tcs:
        job_id = tc.stats.job_id
        name = tc.stats.name
        serialized = pickle.dumps(tc)
        notes = tc.stats.notes
        uuid = tc.id

        records.append((job_id, name, serialized, notes, uuid))

    command = ''' UPDATE Circs
                    SET jobID=?,
                        name=?,
                        object=?,
                        notes=?
                    WHERE uuid=?'''
    try:
        with sqlite3.connect(db) as conn:
            conn.executemany(command, records)
    except sqlite3.DatabaseError:
        logger.error(f'Database update error on Table Circs: {traceback.format_exc()}')
    finally:
        if conn: conn.close()


def retrieve_objects(db: str, ids: List[str]) -> List[TestCircuit]:
    conformed_list = [(uuid,) for uuid in ids]
    circs = []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            command = '''SELECT uuid, object
                            FROM Circs
                            WHERE uuid=?'''
            for uuid in conformed_list:
                row = conn.execute(command, uuid).fetchone()
                tc = pickle.loads(row['object'])
                circs.append(tc)
    except sqlite3.DatabaseError:
        msg = 'Error retrieving objects from Circs'
        error = traceback.format_exc()
        logger.error(f'{msg}: {error}')
    finally:
        if conn: conn.close()

    return circs


def record_exists(db: str, uuid: str) -> bool:
    check = '''SELECT uuid
                FROM CIRCS
                WHERE uuid=?'''

    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(check, (uuid,))
            if conn.fetchone()[0]:
                return True
            else:
                return False
    except sqlite3.DatabaseError:
        logger.error(f'Database retrieval error on Table Circs: {traceback.format_exc()}')
    finally:
        if conn: conn.close()


def write_stats(db: str, ids: List[str]) -> None:
    conformed_list = [(uuid,) for uuid in ids]
    circs = []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            command = '''SELECT uuid, object
                            FROM Circs
                            WHERE uuid=?'''

            for uuid in conformed_list:
                row = conn.execute(command, uuid).fetchone()
                tc = pickle.loads(row['object'])
                circs.append(tc)

            stats_command = '''INSERT INTO Stats(id, name, backend, truth_value, ideal, results, circ_width, pre_depth, 
            post_depth, swap_count, compile_time, datetime, batch_avg, global_avg, job_id, notes)
                                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

            records = []
            for tc in circs:
                stats: Statblock = tc.stats
                row_values = (stats.id,
                              stats.name,
                              stats.backend,
                              stats.truth_value,
                              str(stats.ideal_distribution),
                              str(stats.results),
                              stats.circ_width,
                              stats.pre_depth,
                              stats.post_depth,
                              stats.swap_count,
                              stats.compile_time,
                              stats.datetime,
                              stats.batch_avg,
                              stats.global_avg,
                              stats.job_id,
                              stats.notes)

                records.append(row_values)

            conn.executemany(stats_command, records)

    except sqlite3.DatabaseError:
        msg = 'Error retrieving and inserting all stats'
        error = traceback.format_exc()
        logger.error(f'{msg}: {error}')
    finally:
        if conn: conn.close()


def write_all_stats(db: str) -> None:
    circs = []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            command = '''SELECT uuid, object
                            FROM Circs'''

            for record in conn.execute(command):
                tc = pickle.loads(record['object'])
                circs.append(tc)

            stats_command = '''INSERT INTO Stats(id, name, backend, truth_value, ideal, results, circ_width, pre_depth, 
            post_depth, swap_count, compile_time, datetime, batch_avg, global_avg, job_id, notes)
                                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

            records = []
            for tc in circs:
                stats: Statblock = tc.stats
                row_values = (stats.id,
                              stats.name,
                              stats.backend,
                              stats.truth_value,
                              str(stats.ideal_distribution),
                              str(stats.results),
                              stats.circ_width,
                              stats.pre_depth,
                              stats.post_depth,
                              stats.swap_count,
                              stats.compile_time,
                              stats.datetime,
                              stats.batch_avg,
                              stats.global_avg,
                              stats.job_id,
                              stats.notes)

                records.append(row_values)

            conn.executemany(stats_command, records)

    except sqlite3.DatabaseError:
        msg = 'Error retrieving and inserting all stats'
        error = traceback.format_exc()
        logger.error(f'{msg}: {error}')
    finally:
        if conn: conn.close()


def is_empty(db: str, table_name: str) -> bool:
    check = f'''SELECT count(*)
                FROM (SELECT 0 FROM {table_name} LIMIT 1)'''
    try:
        with sqlite3.connect(db) as conn:
            c = conn.cursor()
            c.execute(check)
            if c.fetchone()[0]:
                return False
            else:
                return True
    except sqlite3.DatabaseError:
        logger.error(f'Error counting database records: {traceback.format_exc()}')
    finally:
        if conn: conn.close()


def create_all_tables(db: str = db_location) -> None:
    """ Conditionally creates db file at db_location if it does not exist, and conditionally creates all tables
        if they do not exist.  If everything already exists, no changes are made.

    Args:
        db (str): Location to create the db at. If no path and/or name were passed when the experiment was run, defaults
            to 'circuit_data.sqlite'

    Returns: None

    """
    create_circs = '''create table if not exists Circs
    (
        uuid text
            constraint Circs_pk
                primary key,
        jobID TEXT,
        name TEXT,
        object BLOB,
        notes TEXT
    );'''

    create_running = '''create table if not exists Running
    (
        uuid TEXT
            constraint Running_pk
                primary key,
        jobID TEXT,
        name TEXT,
        object TEXT
    );'''

    create_running_index = '''create unique index if not exists Running_uuid_uindex on Running(uuid);'''

    create_stats = '''create table if not exists Stats
    (
        id TEXT,
        name TEXT,
        backend TEXT,
        truth_value int,
        ideal TEXT,
        results TEXT,
        circ_width int,
        pre_depth int,
        post_depth int,
        swap_count int,
        compile_time REAL,
        datetime TEXT,
        batch_avg REAL,
        global_avg REAL,
        job_id TEXT,
        notes TEXT,
        edistance REAL
    );'''

    # Creating a connection object to db_location will create a db file if none exists.
    try:
        logger.info(f'Connecting to db file (creating if it does not exist...)')
        with sqlite3.connect(db_location) as conn:
            logger.info(f'Creating Circs table ...')
            conn.execute(create_circs)
            logger.info(f'Circs table created!')
            logger.info(f'Creating Running table ...')
            conn.execute(create_running)
            conn.execute(create_running_index)
            logger.info(f'Running table created!')
            logger.info(f'Creating Stats table ...')
            conn.execute(create_stats)
            logger.info(f'Stats table created!')

    except sqlite3.DatabaseError:
        msg = f'Error creating db tables for db at {db_location}'
        error = traceback.format_exc()
        logger.critical(f'{msg}: {error}')

    finally:
        if conn: conn.close()
