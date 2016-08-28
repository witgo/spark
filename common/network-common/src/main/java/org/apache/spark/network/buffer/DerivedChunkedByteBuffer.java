package org.apache.spark.network.buffer;

import java.nio.ByteBuffer;

public class DerivedChunkedByteBuffer extends ChunkedByteBufferImpl {

  final ChunkedByteBuffer unwrap;

  public DerivedChunkedByteBuffer(ByteBuffer[] chunks, ChunkedByteBuffer unwrap) {
    super(chunks);
    this.unwrap = unwrap;
  }

//  public DerivedChunkedByteBuffer() {
//    super();
//    this.unwrap = this;
//  }

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
