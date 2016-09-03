package org.apache.spark.network.buffer.netty;

import io.netty.buffer.ByteBuf;
import org.apache.spark.network.buffer.ChunkedByteBuffer;

public class DerivedChunkedByteBuffer extends ChunkedByteBufImpl {

  final ChunkedByteBuffer unwrap;

  public DerivedChunkedByteBuffer(ByteBuf[] chunks, ChunkedByteBuffer unwrap) {
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
