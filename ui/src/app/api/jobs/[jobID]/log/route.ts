import { NextRequest } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { encryptedJsonResponse } from '@/utils/serverEncryption';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return encryptedJsonResponse({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'log.txt');

  if (!fs.existsSync(logPath)) {
    return encryptedJsonResponse({ log: '' });
  }
  let log = '';
  try {
    log = fs.readFileSync(logPath, 'utf-8');
  } catch (error) {
    console.error('Error reading log file:', error);
    log = 'Error reading log file';
  }
  return encryptedJsonResponse({ log: log });
}
