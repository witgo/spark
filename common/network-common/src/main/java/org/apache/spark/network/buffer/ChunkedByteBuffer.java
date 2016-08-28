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

import java.io.Externalizable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;

import io.netty.buffer.ByteBuf;


public interface ChunkedByteBuffer extends Externalizable, ReferenceCounted {

  /**
   * This size of this buffer, in bytes.
   */
  long size();

  /**
   * Write this buffer to a channel.
   */
  void writeFully(WritableByteChannel channel) throws IOException;

  /**
   * Wrap this buffer to view it as a Netty ByteBuf.
   */
  ByteBuf toNetty();

  /**
   * Copy this buffer into a new byte array.
   *
   * @throws UnsupportedOperationException if this buffer's size exceeds the maximum array size.
   */
  byte[] toArray();

  /**
   * Copy this buffer into a new ByteBuffer.
   *
   * @throws UnsupportedOperationException if this buffer's size exceeds the max ByteBuffer size.
   */
  ByteBuffer toByteBuffer();

  ChunkedByteBufferInputStream toInputStream();

  /**
   * Creates an input stream to read data from this ChunkedByteBuffer.
   *
   * @param dispose if true, [[dispose()]] will be called at the end of the stream
   *                in order to close any memory-mapped files which back this buffer.
   */
  ChunkedByteBufferInputStream toInputStream(boolean dispose);

  /**
   * Make a copy of this ChunkedByteBuffer, copying all of the backing data into new buffers.
   * The new buffer will share no resources with the original buffer.
   *
   * @param allocator a method for allocating byte buffers
   */
  ChunkedByteBuffer copy(Allocator allocator);

  /**
   * Get duplicates of the ByteBuffers backing this ChunkedByteBuffer.
   */
  ByteBuffer[] getChunks();


  ChunkedByteBuffer slice(long offset, long length);

  ChunkedByteBuffer duplicate();

  ChunkedByteBuffer retain();

  ChunkedByteBuffer retain(int increment);
}
