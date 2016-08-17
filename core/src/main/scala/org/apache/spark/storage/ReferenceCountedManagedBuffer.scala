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

package org.apache.spark.storage

import java.io.InputStream

import io.netty.util.{AbstractReferenceCounted, IllegalReferenceCountException}

import org.apache.spark.network.buffer.{ChunkedByteBuffer, ManagedBuffer, NettyManagedBuffer}

private[storage] class ReferenceCountedManagedBuffer(
  var managedBuffer: ManagedBuffer, val onDeallocate: () => Unit) extends ManagedBuffer {
  def this(chunkedBuffer: ChunkedByteBuffer, onDeallocate: () => Unit) {
    this(new NettyManagedBuffer(chunkedBuffer.toNetty), onDeallocate)
  }

  def size: Long = {
    ensureAccessible()
    managedBuffer.size()
  }

  def nioByteBuffer: ChunkedByteBuffer = {
    ensureAccessible()
    managedBuffer.nioByteBuffer()
  }

  def createInputStream: InputStream = {
    ensureAccessible()
    managedBuffer.createInputStream()
  }

  def retain: ManagedBuffer = {
    referenceCounter.retain()
    this
  }

  def release: ManagedBuffer = {
    referenceCounter.release()
    this
  }

  def convertToNetty: AnyRef = {
    ensureAccessible()
    managedBuffer.convertToNetty()
  }

  def refCnt: Int = referenceCounter.refCnt()

  private val referenceCounter = new AbstractReferenceCounted {
    override def deallocate(): Unit = {
      onDeallocate()
    }
  }

  private def ensureAccessible() {
    if (refCnt == 0) throw new IllegalReferenceCountException(0)
  }
}
