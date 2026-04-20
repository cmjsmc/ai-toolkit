import { PrismaClient } from '@prisma/client';
import { isMac } from '@/helpers/basic';
import { getDecryptedJson, encryptedJsonResponse } from '@/utils/serverEncryption';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return encryptedJsonResponse(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return encryptedJsonResponse(job);
    }

    const jobs = await prisma.job.findMany({
      where: job_type ? { job_type } : undefined,
      orderBy: { created_at: 'desc' },
    });
    return encryptedJsonResponse({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return encryptedJsonResponse({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await getDecryptedJson(request);
    const { id, name, job_config } = body;
    let gpu_ids: string = body.gpu_ids;

    if (isMac()) {
      gpu_ids = "mps";
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }

    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id) {
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          ...extra,
        },
      });
      return encryptedJsonResponse(training);
    } else {
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          ...extra,
        },
      });
      return encryptedJsonResponse(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      return encryptedJsonResponse({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    return encryptedJsonResponse({ error: 'Failed to save training data' }, { status: 500 });
  }
}
