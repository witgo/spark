/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.network.buffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;

import com.google.common.io.ByteStreams;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.buffer.Unpooled;
import io.netty.buffer.UnpooledByteBufAllocator;
import io.netty.util.ResourceLeakDetector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.nio.ch.DirectBuffer;

public class ChunkedByteBufferUtil {
  private static final Logger logger = LoggerFactory.getLogger(ChunkedByteBufferUtil.class);

  public static void dispose(ByteBuffer buffer) {
    if (buffer != null && buffer instanceof MappedByteBuffer) {
      logger.trace("Unmapping" + buffer);
      if (buffer instanceof DirectBuffer) {
        DirectBuffer directBuffer = (DirectBuffer) buffer;
        if (directBuffer.cleaner() != null) directBuffer.cleaner().clean();
      }
    }
  }

  public static ChunkedByteBuffer wrap() {
    return new ChunkedByteBufImpl();
  }

  public static ChunkedByteBuffer wrap(ByteBuffer chunk) {
    ByteBuf[] chunks = new ByteBuf[1];
    chunks[0] = Unpooled.wrappedBuffer(chunk);
    return new ChunkedByteBufImpl(chunks);
  }

  public static ChunkedByteBuffer wrap(ByteBuf chunk) {
    ByteBuf[] chunks = new ByteBuf[1];
    chunks[0] = Unpooled.wrappedBuffer(chunk);
    return wrap(chunks);
  }

  public static ChunkedByteBuffer wrap(ByteBuffer[] chunks) {
    ByteBuf[] byteBufs = new ByteBuf[chunks.length];
    for (int i = 0; i < chunks.length; i++) {
      byteBufs[i] = Unpooled.wrappedBuffer(chunks[i]);
    }
    return wrap(byteBufs);
  }

  public static ChunkedByteBuffer wrap(ByteBuf[] chunks) {
    return new ChunkedByteBufImpl(chunks);
  }

  public static ChunkedByteBuffer wrap(byte[] array) {
    return wrap(array, 0, array.length);
  }

  public static ChunkedByteBuffer wrap(byte[] array, int offset, int length) {
    return wrap(ByteBuffer.wrap(array, offset, length));
  }

  public static ChunkedByteBuffer wrap(
      byte[] bytes, int off, int len, int chunkSize) {
    return wrap(bytes, off, len, chunkSize, ChunkedByteBufferUtil.DEFAULT_ALLOCATOR);
  }

  public static ChunkedByteBuffer wrap(
      byte[] bytes, int off, int len, int chunkSize, Allocator allocator) {
    assert bytes.length >= off + len;
    int numChunk = (int) Math.ceil(((double) len) / chunkSize);
    ByteBuf[] chunks = new ByteBuf[numChunk];
    for (int i = 0; i < numChunk; i++) {
      int bufLen = Math.min(len, chunkSize);
      ByteBuf chunk = allocator.allocate(bufLen);
      chunk.writeBytes(bytes, off, bufLen);
      chunks[i] = chunk;
      off += bufLen;
      len -= bufLen;
    }
    return wrap(chunks);
  }

  public static ChunkedByteBuffer wrap(InputStream in, int chunkSize) throws IOException {
    ChunkedByteBufferOutputStream out = new ChunkedByteBufferOutputStream(chunkSize);
    ByteStreams.copy(in, out);
    out.close();
    return out.toChunkedByteBuffer();
  }

  public static ChunkedByteBuffer wrap(InputStream in) throws IOException {
    return wrap(in, 32 * 1024);
  }

  public static ChunkedByteBuffer allocate(int capacity) {
    return allocate(capacity, ChunkedByteBufferUtil.DEFAULT_ALLOCATOR);
  }

  public static ChunkedByteBuffer allocate(int capacity, Allocator allocator) {
    return wrap(allocator.allocate(capacity));
  }

  public static Allocator DEFAULT_ALLOCATOR = new Allocator() {
    @Override
    public ByteBuf allocate(int len) {
      ByteBuf byteBuf = Unpooled.wrappedBuffer(ByteBuffer.allocate(len));
      byteBuf.clear();
      return byteBuf;
//  if (ResourceLeakDetector.getLevel() != ResourceLeakDetector.Level.ADVANCED) {
//    ResourceLeakDetector.setLevel(ResourceLeakDetector.Level.ADVANCED);
//  }
//  return PooledByteBufAllocator.DEFAULT.heapBuffer(len, len);
    }
  };
}
