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

import io.netty.util.internal.PlatformDependent;

import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

/**
 * Abstract base class for classes wants to implement {@link ReferenceCounted}.
 */
public abstract class AbstractReferenceCounted implements ReferenceCounted {

  private static final AtomicIntegerFieldUpdater<AbstractReferenceCounted> refCntUpdater;

  static {
    AtomicIntegerFieldUpdater<AbstractReferenceCounted> updater =
        PlatformDependent.newAtomicIntegerFieldUpdater(AbstractReferenceCounted.class, "refCnt");
    if (updater == null) {
      updater = AtomicIntegerFieldUpdater.newUpdater(AbstractReferenceCounted.class, "refCnt");
    }
    refCntUpdater = updater;
  }

  private volatile int refCnt = 1;

  @Override
  public int refCnt() {
    return refCnt;
  }

  /**
   * An unsafe operation intended for use by a subclass that sets the reference count of the buffer directly
   */
  protected void setRefCnt(int refCnt) {
    this.refCnt = refCnt;
  }

  @Override
  public ReferenceCounted retain() {
    for (; ; ) {
      int refCnt = this.refCnt;
      if (refCnt == 0 || refCnt == Integer.MAX_VALUE) {
        throw new IllegalReferenceCountException(refCnt, 1);
      }
      if (refCntUpdater.compareAndSet(this, refCnt, refCnt + 1)) {
        break;
      }
    }
    return this;
  }

  @Override
  public ReferenceCounted retain(int increment) {
    if (increment <= 0) {
      throw new IllegalArgumentException("increment: " + increment + " (expected: > 0)");
    }

    for (; ; ) {
      final int nextCnt;
      int refCnt = this.refCnt;
      if (refCnt == 0 || (nextCnt = refCnt + increment) < 0) {
        throw new IllegalReferenceCountException(refCnt, increment);
      }
      if (refCntUpdater.compareAndSet(this, refCnt, nextCnt)) {
        break;
      }
    }
    return this;
  }

  @Override
  public boolean release() {
    for (; ; ) {
      int refCnt = this.refCnt;
      if (refCnt == 0) {
        throw new IllegalReferenceCountException(0, -1);
      }

      if (refCntUpdater.compareAndSet(this, refCnt, refCnt - 1)) {
        if (refCnt == 1) {
          deallocate();
          return true;
        }
        return false;
      }
    }
  }

  @Override
  public boolean release(int decrement) {
    if (decrement <= 0) {
      throw new IllegalArgumentException("decrement: " + decrement + " (expected: > 0)");
    }

    for (; ; ) {
      int refCnt = this.refCnt;
      if (refCnt < decrement) {
        throw new IllegalReferenceCountException(refCnt, -decrement);
      }

      if (refCntUpdater.compareAndSet(this, refCnt, refCnt - decrement)) {
        if (refCnt == decrement) {
          deallocate();
          return true;
        }
        return false;
      }
    }
  }

  /**
   * Called once {@link #refCnt()} is equals 0.
   */
  protected abstract void deallocate();
}
