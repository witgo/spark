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

import io.netty.util.AbstractReferenceCounted;
import io.netty.util.IllegalReferenceCountException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class InputStreamManagedBuffer extends ManagedBuffer {
  private final InputStream inputStream;
  private final long byteCount;

  private final AbstractReferenceCounted referenceCounter = new AbstractReferenceCounted() {
    @Override
    protected void deallocate() {
      try {
        inputStream.close();
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
    }
  };

  public InputStreamManagedBuffer(InputStream inputStream, long byteCount) {
    this.inputStream = inputStream;
    this.byteCount = byteCount;
  }

  public long size() {
    return byteCount;
  }

  public ByteBuffer nioByteBuffer() throws IOException {
    throw new UnsupportedOperationException("nioByteBuffer");
  }

  public InputStream createInputStream() throws IOException {
    if (referenceCounter.refCnt() == 0) throw new IllegalReferenceCountException(0);
    return inputStream;
  }

  public ManagedBuffer retain() {
    referenceCounter.retain();
    return this;
  }

  public ManagedBuffer release() {
    referenceCounter.release();
    return this;
  }

  public Object convertToNetty() throws IOException {
    throw new UnsupportedOperationException("convertToNetty");
  }
}
