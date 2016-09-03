package org.apache.spark.network.buffer.nio;

import org.apache.spark.network.buffer.ChunkedByteBuffer;

import java.nio.ByteBuffer;

public class DerivedChunkedByteBuffer extends ChunkedByteBufferImpl {

  final ChunkedByteBuffer unwrap;

  public DerivedChunkedByteBuffer(ByteBuffer[] chunks, ChunkedByteBuffer unwrap) {
    super(chunks);
    this.unwrap = unwrap;
  }

  @Override
  public int refCnt() {
    return unwrap.refCnt();
  }

  @Override
  public DerivedChunkedByteBuffer retain() {
    unwrap.retain();
    return this;
  }

  @Override
  public DerivedChunkedByteBuffer retain(int increment) {
    unwrap.retain();
    return this;
  }

  @Override
  public boolean release() {
    return unwrap.release();
  }

  @Override
  public boolean release(int decrement) {
    return unwrap.release();
  }
}
