import argparse
from asyncio import tasks
import asyncio
import enum
import json
import logging
import queue
import threading
import time
import typing
import time_machine as tm
import requests
import yaml
import get_subdomain as pdns
import datetime
import os

global_alive_done = False

def task_get_subdomains(
    pdns_domains: typing.Set,
    result_dir: str,
    pdns_token: str,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    thread_count: int=10,
):
    datetime_format = "%Y%m%d%H%M%S"
    # TODO may split the time windows into smaller pieces to increase throughput
    time_windows = [(
        start_datetime.strftime(datetime_format),
        end_datetime.strftime(datetime_format)
    )]
    domain_queue = queue.Queue()
    for domain in pdns_domains:
        for tw_start, tw_end in time_windows:
            domain_queue.put((domain, tw_start, tw_end))   
    if configs["pdns"]["thread_count"]  > domain_queue.qsize():
        thread_count = domain_queue.qsize() 
    threads = []
    for i in range(thread_count):
        thread = pdns.PDNSThread(
            domain_queue=domain_queue,
            pdns_token=pdns_token,
            result_dir=result_dir,
        )
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with open(os.path.join(result_dir, "unique_domains.txt"), "w") as fd:
        for domain in pdns.global_unique_domains:
            fd.write(f"{domain}\n")
    end_time = time.time()
    logging.info(
        "End with %d domains queried,and  a time cost of %d seconds",
        len(pdns_domains),
        end_time - start_time,
    )
    logging.info(
        "End with %d pdns records captured, and %d unique domains observed",
        pdns.global_pdns_count,
        len(pdns.global_unique_domains),
    )


class AliveThread(threading.Thread):
    def __init__(
        self,
        task_queue: queue.Queue,
        result_queue: queue.Queue,
        timeout: float=10,
    ) ->None:
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.timeout = timeout

    def run(self) ->None:
        requests.packages.urllib3.disable_warnings()
        while not self.task_queue.empty():
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
            }
            domain = None
            try:
                domain = self.task_queue.get_nowait()
            except Exception as e:
                continue
            result = {}
            is_alive = False
            for scheme in ["http", "https"]:
                url = f"{scheme}://{domain}"
                request_start_time = time.time()
                try:
                    resp = requests.get(url, headers=headers, timeout=self.timeout, verify=False, stream=True)
                    resp_body = ""
                    for chunk in resp.iter_content(chunk_size=512, decode_unicode=False):
                        resp_body = chunk
                        break
                    if type(resp_body) is bytes:
                        resp_body = resp_body.decode(encoding="utf-8", errors="ignore")
                    # logging.info("resp size is %d", len(resp_body))
                    result[scheme] = {
                        "is_exception": False,
                        "status_code": resp.status_code,
                        "resp_body_100": resp_body[:200],
                        "time_cost_in_secs": time.time() - request_start_time,
                    }
                    is_alive = True
                except Exception as e:
                    result[scheme] = {
                        "is_exception": True,
                        "exception_msg": f"{e}",
                        "time_cost_in_secs": time.time() - request_start_time,
                    }
                    logging.debug(f"Got exception whe testing alive for {domain}: {e}")
            result["domain"] = domain
            result["is_alive"] = is_alive
            result["timestamp"] = time.time()
            self.result_queue.put(json.dumps(result))


class AliveLogThread(threading.Thread):
    def __init__(self, result_file, result_queue: queue.Queue) ->None:
        super().__init__()
        self.result_file = result_file
        self.result_queue = result_queue

    def run(self) ->None:
        log_count = 0
        start_time = time.time()
        cache = []
        with open(self.result_file, "a") as fd:
            while True:
                if self.result_queue.empty():
                    if global_alive_done:
                        break
                    time.sleep(1)
                    continue
                # Since this thread is the only consumer of the result queue, no race condition for the result items
                result_item = self.result_queue.get()
                cache.append(result_item)
                if len(cache) >= 100:
                    fd.write("\n".join(cache) + "\n")
                    fd.flush()
                    log_count += len(cache)
                    cache = []
                    logging.info(
                        f"Finish ping tests for {log_count} domains"
                        f" with time cost of {int(time.time() - start_time)} seconds"
                        f" with {self.result_queue.qsize()} left to dump"
                    )
            if len(cache) > 0:
                fd.write("\n".join(cache) + "\n")
            


def task_test_alive(
    domains: typing.Set,
    result_file: str,
    thread_count: int=10,
    timeout: float=10,
) ->None:
    """ Test aliveness of the given domains
    """
    global global_alive_done
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    for domain in domains:
        task_queue.put(domain)
    if thread_count > task_queue.qsize():
        thread_count = task_queue.qsize()
    alive_test_threads = [
        AliveThread(task_queue=task_queue, result_queue=result_queue, timeout=timeout)
        for i in range(thread_count)
    ]
    for thread in alive_test_threads:
        thread.start()
    log_thread = AliveLogThread(
        result_file=result_file,
        result_queue=result_queue,
    )
    log_thread.start()
    for thread in alive_test_threads:
        thread.join()
    # send the following signal to the log thread
    global_alive_done = True
    log_thread.join()
    
def ngrok_if_screenshot(alive_test_result: typing.Dict[str, any]) ->bool:
    """Decide from the alive test result, whether a given domain should be screenshot""" 
    is_exception = True
    is_not_found = True
    # the following text is part of the description tag in the html head
    error_page_signature = "ngrok is the fastest way"
    for scheme in ["http", "https"]:
        resp = alive_test_result[scheme]
        if resp["is_exception"]:
            continue
        is_exception = False
        if (
            resp["status_code"] != 404 
            or 
            error_page_signature not in resp["resp_body_100"]
        ):
            is_not_found = False
    if is_exception or is_not_found:
        return False
    return True

def oray_if_screenshot(alive_test_result: typing.Dict[str, any]) ->bool:
    """Decide from the alive test result, whether a given domain should be screenshot""" 
    is_exception = True
    is_not_found = False
    not_found_messages = [
        "很抱歉，您访问的花生壳动态域名不在线",
        "您的域名已被锁定",
        "\xe6\x82\xa8\xe7\x9a\x84\xe5\x9f\x9f\xe5",
        "花生壳过期页面",
        "花生壳封锁页面",
    ]
    for scheme in ["http", "https"]:
        resp = alive_test_result[scheme]
        if resp["is_exception"]:
            continue
        # as long as one resp is not an exception
        is_exception = False
        resp_body = resp["resp_body_100"]
        for msg in not_found_messages:
            # As long as the resp matches one of the not-found messages
            if msg in resp_body:
                is_not_found = True
    if is_exception or is_not_found:
        return False
    return True

def get_alive_domains_to_screenshot(
    alive_test_file: str,
    provider: str,
) ->typing.Set[str]:
    result_domains = set()
    if provider == "ngrok":
        if_screenshot = ngrok_if_screenshot
    elif provider == "oray":
        if_screenshot = oray_if_screenshot
    else:
        raise Exception(f"Unknown provider {provider}")
    with open(alive_test_file, "r") as fd:
        for line in fd:
            alive_test_result = json.loads(line)
            if if_screenshot(alive_test_result):
                result_domains.add(alive_test_result["domain"])
    return result_domains



def task_screenshot(
    domains: typing.Set,
    result_dir: str,
    provider: str,
    process_count: int=5,
    enable_multi_screen=False,
    multi_screen_limit=10,
):
    asyncio.run(
        tm.screen_webpages(
            domains,
            result_dir,
            provider,
            batch_size=process_count,
            enable=enable_multi_screen,
            limit=multi_screen_limit,
        )
    )

def init_data_directories(configs):
    result_configs = {}
    result_configs.update(configs)
    if configs["date_tag"] is not None:
        date_str = configs["date_tag"]
    else:
        date_str = datetime.date.today().strftime("%Y%m%d")
    pdns_result_dir = os.path.join(configs["data_dir"], "subdomains", date_str)
    if not os.path.exists(pdns_result_dir):
        os.makedirs(pdns_result_dir)
    result_configs["pdns_result_dir"] = pdns_result_dir
    # alive directory
    alive_result_dir = os.path.join(configs["data_dir"], "alive_test", date_str)
    if not os.path.exists(alive_result_dir):
        os.makedirs(alive_result_dir)
    result_configs["alive_result_dir"] = alive_result_dir
    # screenshot
    screenshot_result_dir = os.path.join(configs["data_dir"], "screenshots", date_str)
    if not os.path.exists(screenshot_result_dir):
        os.makedirs(screenshot_result_dir)
    result_configs["screenshot_result_dir"] = screenshot_result_dir
    return result_configs

class Step(enum.Enum):
    PDNS = 1
    ALIVE = 2
    SCREENSHOT = 4

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(threadName)s %(asctime)s %(levelname)s %(message)s")
    start_time = time.time()
    parser = argparse.ArgumentParser("The pipeline to collect and screenshot PFS domains")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--data_dir", "-dd", type=str, default=None)
    parser.add_argument("--domain_file", "-df", type=str, default=None)
    parser.add_argument("--alive_seed_file", "-asf", type=str, default=None)
    parser.add_argument("--steps", "-ss", default=7, type=int)
    parser.add_argument("--enable_multi_screen", "-ems", action="store_true")
    parser.add_argument("--multi_screen_limit", "-msl", type=int, default=10)
    options = parser.parse_args()
    with open(options.config_file, "r") as f:
        configs = yaml.safe_load(f) 
    if options.data_dir:
        configs["data_dir"] = options.data_dir
    if options.domain_file:
        configs["pdns"]["domain_file"] = options.domain_file
    configs = init_data_directories(configs)
    # print(configs)
    # assert False
    # collect latest PFS subdomains
    end_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
    start_datetime = end_datetime - datetime.timedelta(days=configs["pdns"]["time_delta_in_days"])
    if options.steps & Step.PDNS.value:
        logging.info("Crawl PFS subdomains")
        task_get_subdomains(
            pdns_domains=set(open(configs["pdns"]["domain_file"], "r").read().splitlines()),
            result_dir=configs["pdns_result_dir"],
            pdns_token=configs["pdns"]["access_token"],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            thread_count=configs["pdns"]["thread_count"],
        )   
    # test aliveness of PFS subdomains
    # TODO optionally load from the global variable of last step instead of the local file
    alive_result_file = os.path.join(configs["alive_result_dir"], "alive_test_results.json")
    if options.steps & Step.ALIVE.value:
        domains_for_alive_test = set()
        # By default, it reads unique domains retrieved from pdns queries, unless specified in command line args
        alive_seed_file = os.path.join(configs["pdns_result_dir"], "unique_domains.txt")
        if options.alive_seed_file:
            alive_seed_file = options.alive_seed_file
        with open(alive_seed_file, "r") as fd:
            for line in fd:
                domains_for_alive_test.add(line.strip())
        logging.info(
            f"Test aliveness of {len(domains_for_alive_test)}"
            " PFS subdomains as captured in last step"
        )
        task_test_alive(
            domains=domains_for_alive_test,
            result_file=alive_result_file,
            thread_count=configs["alive"]["thread_count"],
            timeout=configs["alive"]["timeout"],
        )   

    # screenshot alive PFS subdomains
    if options.steps & Step.SCREENSHOT.value:
        logging.info("Screenshot visitable PFS subdomains")
        domains_to_screenshot = get_alive_domains_to_screenshot(
            alive_result_file,
            provider=configs["provider"],
        )
        logging.info(f"Got {len(domains_to_screenshot)} domains to screenshot")
        task_screenshot(
            domains=domains_to_screenshot,
            provider=configs["provider"],
            result_dir=configs["screenshot_result_dir"],
            process_count=configs["screenshot"]["process_count"],
            enable_multi_screen=options.enable_multi_screen,
            multi_screen_limit=options.multi_screen_limit,
        )
    logging.info(
        "Done all the tasks with time cost %d secs",
        time.time() - start_time,
    )
