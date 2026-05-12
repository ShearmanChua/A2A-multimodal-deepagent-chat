/**
 * S3-compatible object store upload utility.
 *
 * Configuration via environment variables (mirrors the Python side):
 *   OBJECT_STORE_ENDPOINT          – host:port, e.g. "seaweedfs:8333" (no scheme)
 *   OBJECT_STORE_ACCESS_KEY        – access key (default "")
 *   OBJECT_STORE_SECRET_KEY        – secret key (default "")
 *   OBJECT_STORE_SECURE            – "true" for HTTPS (default "false")
 *   OBJECT_STORE_BUCKET            – bucket for media uploads (default "media")
 *   OBJECT_STORE_PRESIGN_EXPIRY    – pre-signed URL TTL in seconds (default 3600)
 *   OBJECT_STORE_EXTERNAL_ENDPOINT – external host for pre-signed URLs when the
 *       internal endpoint is not reachable from clients
 *       (e.g. "localhost:8333" when running inside Docker)
 */

const { S3Client, PutObjectCommand, GetObjectCommand, CreateBucketCommand, HeadBucketCommand } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const path = require("path");
const { v4: uuidv4 } = require("uuid");

function _makeClient(endpoint) {
  const secure = ["1", "true", "yes"].includes(
    (process.env.OBJECT_STORE_SECURE || "false").toLowerCase()
  );
  const scheme = secure ? "https" : "http";
  return new S3Client({
    endpoint: `${scheme}://${endpoint}`,
    region: "us-east-1",
    credentials: {
      accessKeyId: process.env.OBJECT_STORE_ACCESS_KEY || "",
      secretAccessKey: process.env.OBJECT_STORE_SECRET_KEY || "",
    },
    forcePathStyle: true,
  });
}

function _uploadClient() {
  return _makeClient(process.env.OBJECT_STORE_ENDPOINT || "seaweedfs:8333");
}

function _presignClient() {
  const external = process.env.OBJECT_STORE_EXTERNAL_ENDPOINT;
  return _makeClient(external || process.env.OBJECT_STORE_ENDPOINT || "seaweedfs:8333");
}

/** Return true when an object store endpoint is configured. */
function isObjectStoreAvailable() {
  return !!process.env.OBJECT_STORE_ENDPOINT;
}

async function _ensureBucket(client, bucket) {
  try {
    await client.send(new HeadBucketCommand({ Bucket: bucket }));
  } catch {
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
  }
}

/**
 * Upload a Buffer to the object store and return a pre-signed GET URL.
 *
 * @param {Buffer} buffer       - File bytes.
 * @param {string} mimeType     - Content-Type.
 * @param {string} originalName - Original filename (used to derive the extension).
 * @returns {Promise<string>}   Pre-signed URL.
 */
async function uploadBufferAndPresign(buffer, mimeType, originalName) {
  const bucket = process.env.OBJECT_STORE_BUCKET || "media";
  const expiry = parseInt(process.env.OBJECT_STORE_PRESIGN_EXPIRY || "3600", 10);
  const ext = path.extname(originalName).toLowerCase() || ".bin";
  const key = `uploads/${Date.now()}_${uuidv4().slice(0, 8)}${ext}`;

  const uploadClient = _uploadClient();
  await _ensureBucket(uploadClient, bucket);

  await uploadClient.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: buffer,
      ContentType: mimeType,
      ContentLength: buffer.length,
    })
  );

  const url = await getSignedUrl(
    _presignClient(),
    new GetObjectCommand({ Bucket: bucket, Key: key }),
    { expiresIn: expiry }
  );

  console.log(`[objectStore] Uploaded ${buffer.length} bytes → ${key} (expiry=${expiry}s)`);
  return url;
}

module.exports = { isObjectStoreAvailable, uploadBufferAndPresign };
