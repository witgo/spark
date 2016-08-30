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
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;
import java.util.ArrayList;
import java.util.Arrays;

import com.google.common.base.Objects;
import com.google.common.base.Throwables;
import com.google.common.base.Preconditions;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.spark.network.util.ByteArrayWritableChannel;
import org.apache.spark.network.util.JavaUtils;

public class ChunkedByteBufImpl extends AbstractReferenceCounted implements ChunkedByteBuffer {
  private static final Logger logger = LoggerFactory.getLogger(ChunkedByteBufImpl.class);
  private static final int BUF_SIZE = 0x1000; // 4K
  private static final ByteBuf[] emptyChunks = new ByteBuf[0];
  private ByteBuf[] chunks = null;

  // For deserialization only
  public ChunkedByteBufImpl() {
    this(emptyChunks);
  }

  /**
   * Read-only byte buffer which is physically stored as multiple chunks rather than a single
   * contiguous array.
   *
   * @param chunks an array of [[ByteBuffer]]s. Each buffer in this array must have position == 0.
   *               Ownership of these buffers is transferred to the ChunkedByteBuffer, so if these
   *               buffers may also be used elsewhere then the caller is responsible for copying
   *               them as needed.
   */
  public ChunkedByteBufImpl(ByteBuf[] chunks) {
    this.chunks = chunks;
    Preconditions.checkArgument(chunks != null, "chunks must not be null");
  }

  @Override
  public void writeExternal(ObjectOutput out) throws IOException {
    ensureAccessible();
    out.writeInt(chunks.length);
    byte[] buf = null;
    for (int i = 0; i < chunks.length; i++) {
      ByteBuf buffer = chunks[i].duplicate();
      int length = buffer.readableBytes();
      out.writeInt(length);
      if (buffer.hasArray()) {
        out.write(buffer.array(), buffer.arrayOffset() + buffer.readerIndex(), length);
        buffer.readerIndex(buffer.readerIndex() + length);
      } else {
        if (buf == null) buf = new byte[BUF_SIZE];
        while (buffer.isReadable()) {
          int r = Math.min(BUF_SIZE, buffer.readableBytes());
          buffer.readBytes(buf, 0, r);
          out.write(buf, 0, r);
        }
      }
    }
  }

  @Override
  public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
    ByteBuf[] buffers = new ByteBuf[in.readInt()];
    byte[] buf = null;
    for (int i = 0; i < buffers.length; i++) {
      int length = in.readInt();
      ByteBuf buffer = ChunkedByteBufferUtil.DEFAULT_ALLOCATOR.allocate(length);
      if (buffer.hasArray()) {
        in.readFully(buffer.array(), buffer.arrayOffset() + buffer.writerIndex(), length);
        buffer.writerIndex(buffer.writerIndex() + length);
      } else {
        if (buf == null) buf = new byte[BUF_SIZE];
        while (length > 0) {
          int r = Math.min(BUF_SIZE, length);
          in.readFully(buf, 0, r);
          buffer.writeBytes(buf, 0, r);
          length -= r;
        }
      }
      buffers[i] = buffer;
    }
    this.chunks = buffers;
  }

  /**
   * This size of this buffer, in bytes.
   */
  @Override
  public long size() {
    ensureAccessible();
    if (chunks == null) return 0L;
    int i = 0;
    long sum = 0L;
    while (i < chunks.length) {
      sum += chunks[i].readableBytes();
      i++;
    }
    return sum;
  }

  /**
   * Write this buffer to a channel.
   */
  @Override
  public void writeFully(WritableByteChannel channel) throws IOException {
    ensureAccessible();
    for (int i = 0; i < chunks.length; i++) {
      ByteBuffer bytes = chunks[i].nioBuffer();
      while (bytes.remaining() > 0) {
        channel.write(bytes);
      }
    }
  }

  /**
   * Wrap this buffer to view it as a Netty ByteBuf.
   */
  @Override
  public ByteBuf toNetty() {
    ensureAccessible();
    long len = size();
    Preconditions.checkArgument(size() <= Integer.MAX_VALUE,
        "Too large ByteBuf: %s", new Object[]{Long.valueOf(len)});
    return Unpooled.wrappedBuffer(getChunks());
  }

  /**
   * Copy this buffer into a new byte array.
   *
   * @throws UnsupportedOperationException if this buffer's size exceeds the maximum array size.
   */
  @Override
  public byte[] toArray() {
    ensureAccessible();
    try {
      if (chunks.length == 1) {
        return JavaUtils.bufferToArray(chunks[0].nioBuffer());
      } else {
        long len = size();
        if (len >= Integer.MAX_VALUE) {
          throw new UnsupportedOperationException("cannot call toArray because buffer size (" +
              len + " bytes) exceeds maximum array size");
        }
        ByteArrayWritableChannel byteChannel = new ByteArrayWritableChannel((int) len);
        writeFully(byteChannel);
        byteChannel.close();
        return byteChannel.getData();
      }
    } catch (Throwable e) {
      throw Throwables.propagate(e);
    }
  }

  /**
   * Copy this buffer into a new ByteBuffer.
   *
   * @throws UnsupportedOperationException if this buffer's size exceeds the max ByteBuffer size.
   */
  @Override
  public ByteBuffer toByteBuffer() {
    ensureAccessible();
    if (chunks.length == 1) {
      return chunks[0].nioBuffer();
    } else {
      return ByteBuffer.wrap(this.toArray());
    }
  }

  @Override
  public ChunkedByteBufferInputStream toInputStream() {
    return toInputStream(false);
  }

  /**
   * Creates an input stream to read data from this ChunkedByteBuffer.
   *
   * @param dispose if true, [[dispose()]] will be called at the end of the stream
   *                in order to close any memory-mapped files which back this buffer.
   */
  @Override
  public ChunkedByteBufferInputStream toInputStream(boolean dispose) {
    ensureAccessible();
    return new ChunkedByteBufferInputStream(this, dispose);
  }

  /**
   * Make a copy of this ChunkedByteBuffer, copying all of the backing data into new buffers.
   * The new buffer will share no resources with the original buffer.
   *
   * @param allocator a method for allocating byte buffers
   */
  @Override
  public ChunkedByteBuffer copy(Allocator allocator) {
    ensureAccessible();
    ByteBuf[] copiedChunks = new ByteBuf[chunks.length];
    for (int i = 0; i < chunks.length; i++) {
      ByteBuf chunk = chunks[i].duplicate();

      ByteBuf newChunk = allocator.allocate(chunk.readableBytes());
      newChunk.writeBytes(chunk);

      copiedChunks[i] = newChunk;
    }
    return ChunkedByteBufferUtil.wrap(copiedChunks);
  }

  /**
   * Get duplicates of the ByteBuffers backing this ChunkedByteBuffer.
   */
  @Override
  public ByteBuffer[] getChunks() {
    ensureAccessible();
    ByteBuffer[] buffs = new ByteBuffer[chunks.length];
    for (int i = 0; i < chunks.length; i++) {
      buffs[i] = chunks[i].nioBuffer();
    }
    return buffs;
  }

  @Override
  public ChunkedByteBuffer slice(long offset, long length) {
    ensureAccessible();
    long thisSize = size();
    if (offset < 0 || offset > thisSize - length) {
      throw new IndexOutOfBoundsException(String.format(
          "index: %d, length: %d (expected: range(0, %d))", offset, length, thisSize));
    }
    if (length == 0) {
      return ChunkedByteBufferUtil.wrap();
    }
    ArrayList<ByteBuf> list = new ArrayList<>();
    int i = 0;
    long sum = 0L;
    while (i < chunks.length && length > 0) {
      long lastSum = sum + chunks[i].readableBytes();
      if (lastSum > offset) {
        ByteBuf buffer = chunks[i].duplicate();
        int localLength = (int) Math.min(length, buffer.readableBytes());
        if (localLength < buffer.readableBytes()) {
          buffer.slice(0, localLength);
        }
        length -= localLength;
        list.add(buffer);
      }
      sum = lastSum;
      i++;
    }
    return new DerivedChunkedByteBuffer(list.toArray(new ByteBuf[list.size()]), this);
  }

  @Override
  public ChunkedByteBuffer duplicate() {
    ensureAccessible();
    ByteBuf[] buffs = new ByteBuf[chunks.length];
    for (int i = 0; i < chunks.length; i++) {
      buffs[i] = chunks[i].duplicate();
    }
    return new DerivedChunkedByteBuffer(buffs, this);
  }

  @Override
  public ChunkedByteBuffer retain() {
    super.retain();
    return this;
  }

  @Override
  public ChunkedByteBuffer retain(int increment) {
    super.retain(increment);
    return this;
  }

  /**
   * Attempt to clean up a ByteBuffer if it is memory-mapped. This uses an *unsafe* Sun API that
   * might cause errors if one attempts to read from the unmapped buffer, but it's better than
   * waiting for the GC to find it because that could lead to huge numbers of open files. There's
   * unfortunately no standard API to do this.
   */
  @Override
  protected void deallocate() {
    for (int i = 0; i < chunks.length; i++) {
      chunks[i].release();
    }
  }

  /**
   * Should be called by every method that tries to access the buffers content to check
   * if the buffer was released before.
   */
  protected final void ensureAccessible() {
    if (refCnt() == 0) throw new IllegalReferenceCountException(0);
  }

  @Override
  public int hashCode() {
    ensureAccessible();
    return Arrays.hashCode(chunks);
  }

  @Override
  public boolean equals(Object other) {
    ensureAccessible();
    if (other != null && other instanceof ChunkedByteBuffer) {
      ChunkedByteBuffer o = (ChunkedByteBuffer) other;
      return Objects.equal(chunks, chunks);
    }
    return false;
  }
}
