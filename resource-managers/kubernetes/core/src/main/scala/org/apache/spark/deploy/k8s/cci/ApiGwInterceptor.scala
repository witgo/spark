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
package org.apache.spark.deploy.k8s.cci

import scala.collection.JavaConverters._

import com.cloud.apigateway.sdk.utils.Client
import com.cloud.sdk.http.HttpMethodName
import okhttp3.{Interceptor, Response}
import okio.Buffer

class ApiGwInterceptor(ak: String, sk: String) extends Interceptor {

  override def intercept(chain: Interceptor.Chain): Response = {
    val request = chain.request()
    val httpMethod = Enum.valueOf(classOf[HttpMethodName], request.method())
    val url = request.url().toString()
    val headers = request.headers().toMultimap().asScala
      .mapValues(_.asScala.distinct.toArray)
      .mapValues { vals =>
        assert(vals.size == 1, "Repeat header")
        vals.head
      }.asJava
    val body: String = if (request.body() != null) {
      val buffer = new Buffer()
      request.body().writeTo(buffer)
      buffer.readByteString().utf8()
    } else {
      null
    }
    val authReq = Client.okhttpRequest(httpMethod, ak, sk, url, headers, body)
    chain.proceed(authReq)
  }
}
