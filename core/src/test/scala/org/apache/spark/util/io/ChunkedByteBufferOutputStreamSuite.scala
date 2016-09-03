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

package org.apache.spark.util.io

import scala.util.Random

import org.apache.spark.SparkFunSuite
import org.apache.spark.network.buffer.ChunkedByteBufferOutputStream

class ChunkedByteBufferOutputStreamSuite extends SparkFunSuite {

  test("empty output") {
    val o = ChunkedByteBufferOutputStream.newInstance(1024)
    assert(o.toChunkedByteBuffer.size === 0)
  }

  test("write a single byte") {
    val o = ChunkedByteBufferOutputStream.newInstance(1024)
    o.write(10)
    val chunkedByteBuffer = o.toChunkedByteBuffer
    assert(chunkedByteBuffer.toByteBuffers().length === 1)
    assert(chunkedByteBuffer.toByteBuffers().head.remaining() === 1)
    assert(Seq(chunkedByteBuffer.toByteBuffers().head.get()) === Seq(10.toByte))
  }

  test("write a single near boundary") {
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(new Array[Byte](9))
    o.write(99)
    val chunkedByteBuffer = o.toChunkedByteBuffer
    assert(chunkedByteBuffer.toByteBuffers().length === 1)
    assert(chunkedByteBuffer.toByteBuffers().head.get(9) === 99.toByte)
  }

  test("write a single at boundary") {
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(new Array[Byte](10))
    o.write(99)
    val arrays = o.toChunkedByteBuffer.toByteBuffers()
    assert(arrays.length === 2)
    assert(arrays(1).remaining() === 1)
    assert(arrays(1).get() === 99.toByte)
  }

  test("single chunk output") {
    val ref = new Array[Byte](8)
    Random.nextBytes(ref)
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(ref)
    val arrays = o.toChunkedByteBuffer.toByteBuffers()
    assert(arrays.length === 1)
    assert(arrays.head.remaining() === ref.length)
    val arrRef = new Array[Byte](8)
    arrays.head.get(arrRef)
    assert(arrRef === ref.toSeq)
  }

  test("single chunk output at boundary size") {
    val ref = new Array[Byte](10)
    Random.nextBytes(ref)
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(ref)
    val arrays = o.toChunkedByteBuffer.toByteBuffers()
    assert(arrays.length === 1)
    assert(arrays.head.remaining() === ref.length)
    val arrRef = new Array[Byte](10)
    arrays.head.get(arrRef)
    assert(arrRef === ref.toSeq)
  }

  test("multiple chunk output") {
    val ref = new Array[Byte](26)
    Random.nextBytes(ref)
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(ref)
    val arrays = o.toChunkedByteBuffer.toByteBuffers()
    assert(arrays.length === 3)
    assert(arrays(0).remaining === 10)
    assert(arrays(1).remaining === 10)
    assert(arrays(2).remaining === 6)

    val arrRef = new Array[Byte](10)

    arrays(0).get(arrRef)
    assert(arrRef === ref.slice(0, 10))

    arrays(1).get(arrRef)
    assert(arrRef === ref.slice(10, 20))

    arrays(2).get(arrRef, 0, 6)
    assert(arrRef.slice(0, 6).toSeq === ref.slice(20, 26))
  }

  test("multiple chunk output at boundary size") {
    val ref = new Array[Byte](30)
    Random.nextBytes(ref)
    val o = ChunkedByteBufferOutputStream.newInstance(10)
    o.write(ref)
    val arrays = o.toChunkedByteBuffer.toByteBuffers()
    assert(arrays.length === 3)
    assert(arrays(0).remaining() === 10)
    assert(arrays(1).remaining === 10)
    assert(arrays(2).remaining === 10)

    val arrRef = new Array[Byte](10)

    arrays(0).get(arrRef)
    assert(arrRef === ref.slice(0, 10))

    arrays(1).get(arrRef)
    assert(arrRef === ref.slice(10, 20))

    arrays(2).get(arrRef)
    assert(arrRef === ref.slice(20, 30))
  }
}
