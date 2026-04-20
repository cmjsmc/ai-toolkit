import { exec, execSync } from 'child_process';
import { promisify } from 'util';
import { createRequire } from 'module';
import os from 'os';
import { encryptedJsonResponse } from '@/utils/serverEncryption';

const execAsync = promisify(exec);

interface MacGpuResult {
  name: string;
  memUsed: number;
  memTotal: number;
  gpuLoad: number;
  temperature: number;
  fanSpeed: number;
  powerDraw: number;
}

async function getMacGpuInfo(): Promise<MacGpuResult | null> {
  // [Internal identical to original code]
  try {
    const memoryTotal = os.totalmem() / (1024 * 1024);

    let gpuName = 'Apple GPU';
    try {
      const spOut = execSync(
        'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"',
        { encoding: 'utf-8', timeout: 5000 },
      );
      const nameMatch = spOut.match(/Chipset Model:\s*(.+)/);
      const coresMatch = spOut.match(/Total Number of Cores:\s*(\d+)/);
      if (nameMatch) {
        gpuName = nameMatch[1].trim();
        if (coresMatch) {
          gpuName += ` GPU (${coresMatch[1]} cores)`;
        }
      }
    } catch {
      // fallback to generic name
    }

    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = memoryTotal;

    try {
      const nativeRequire = createRequire(import.meta.url);
      const ms = nativeRequire('macstats') as any;

      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }

      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }

      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }

      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    } catch (error) {
      console.warn('macstats not available:', error);
    }

    return { name: gpuName, memUsed, memTotal, gpuLoad, temperature, fanSpeed, powerDraw };
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    if (isMac) {
      const macGpu = await getMacGpuInfo();
      if (macGpu) {
        return encryptedJsonResponse({
          hasNvidiaSmi: false,
          isMac: true,
          gpus: [
            {
              index: 0,
              name: macGpu.name,
              driverVersion: 'macOS',
              temperature: Math.round(macGpu.temperature),
              utilization: {
                gpu: macGpu.gpuLoad,
                memory: macGpu.memTotal > 0 ? Math.round((macGpu.memUsed / macGpu.memTotal) * 100) : 0,
              },
              memory: {
                total: Math.round(macGpu.memTotal),
                free: Math.round(macGpu.memTotal - macGpu.memUsed),
                used: Math.round(macGpu.memUsed),
              },
              power: { draw: macGpu.powerDraw, limit: 0 },
              clocks: { graphics: 0, memory: 0 },
              fan: { speed: macGpu.fanSpeed },
            },
          ],
        });
      }
      return encryptedJsonResponse({
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [],
        error: 'Could not read Mac GPU stats',
      });
    }

    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

    if (!hasNvidiaSmi) {
      return encryptedJsonResponse({
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: 'nvidia-smi not found or not accessible',
      });
    }

    const gpuStats = await getGpuStats(isWindows);

    return encryptedJsonResponse({
      hasNvidiaSmi: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching NVIDIA GPU stats:', error);
    return encryptedJsonResponse(
      {
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      await execAsync('nvidia-smi -L');
    } else {
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0,
        },
      };
    });

  return gpus;
}
