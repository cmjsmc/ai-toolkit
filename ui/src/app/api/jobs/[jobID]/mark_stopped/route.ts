import { NextRequest } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { encryptedJsonResponse } from '@/utils/serverEncryption';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      status: 'stopped',
      info: 'Job stopped',
      pid: null,
    },
  });

  console.log(`Job ${jobID} marked as stopped`);

  return encryptedJsonResponse(job);
}
