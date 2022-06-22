import nvgpu
from .logger import logger

def get_best_gpu_device():
    '''
    This function return the GPU with the greatest amount of free memory
    '''

    gpu_info = nvgpu.gpu_info()
    free_mem = [x['mem_total'] - x['mem_used'] for x in gpu_info]
    best_gpu_index = free_mem.index(max(free_mem))
    best_gpu = int(gpu_info[best_gpu_index]['index'])
    return best_gpu


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        count = 1
        while True:
            try:
                logger.error("Allocating GPU device, try %s..." % count)
                best_gpu = get_best_gpu_device()
                torch.cuda.set_device(best_gpu)
                break
            except:
                count += 1
                time.sleep(5)
                if count == 10:
                    logger.error("Unable to allocate device %s after 10 tries. Aborting execution..." % best_gpu)
                    exit()
                pass
        logger.info('Choosing GPU number %s' % torch.cuda.current_device())
    return device
