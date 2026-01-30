import aioboto3
import uuid
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


class ConversationHandler:
    region: str = "us-east-1"
    conversations_table: str = "Conversations"
    messages_table: str = "ConversationMessages"

    @staticmethod
    async def create_conversation(
        tenant_id: str,
        title: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        if not title:
            title = "New conversation"

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            item = {
                "TenantId": {"S": tenant_id},
                "ConversationId": {"S": conversation_id},
                "Title": {"S": title},
                "Model": {"S": model},
                "CreatedAt": {"S": now},
                "UpdatedAt": {"S": now},
                "MessageCount": {"N": "0"},
            }

            if system_prompt:
                item["SystemPrompt"] = {"S": system_prompt}

            if metadata:
                item["Metadata"] = {"S": json.dumps(metadata)}

            await client.put_item(
                TableName=ConversationHandler.conversations_table,
                Item=item
            )

            # If system prompt provided, add it as first message
            if system_prompt:
                await ConversationHandler.add_message(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    role="system",
                    content=system_prompt
                )

        return {
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
            "title": title,
            "model": model,
            "created_at": now,
            "updated_at": now,
            "message_count": 1 if system_prompt else 0,
            "metadata": metadata
        }

    @staticmethod
    async def get_conversation(tenant_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            response = await client.get_item(
                TableName=ConversationHandler.conversations_table,
                Key={
                    "TenantId": {"S": tenant_id},
                    "ConversationId": {"S": conversation_id}
                }
            )

            item = response.get("Item")
            if not item:
                return None

            return ConversationHandler._deserialize_conversation(item)

    @staticmethod
    async def list_conversations(
        tenant_id: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
        """List conversations for a tenant, ordered by UpdatedAt (most recent first)"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            query_kwargs = {
                "TableName": ConversationHandler.conversations_table,
                "IndexName": "TenantUpdatedIndex",
                "KeyConditionExpression": "TenantId = :tid",
                "ExpressionAttributeValues": {
                    ":tid": {"S": tenant_id}
                },
                "ScanIndexForward": False,
                "Limit": limit
            }

            if cursor:
                try:
                    query_kwargs["ExclusiveStartKey"] = json.loads(
                        base64.b64decode(cursor).decode()
                    )
                except Exception:
                    pass

            response = await client.query(**query_kwargs)

            items = [
                ConversationHandler._deserialize_conversation(item)
                for item in response.get("Items", [])
            ]

            next_cursor = None
            if "LastEvaluatedKey" in response:
                next_cursor = base64.b64encode(
                    json.dumps(response["LastEvaluatedKey"]).encode()
                ).decode()

            # Get total count
            count_response = await client.query(
                TableName=ConversationHandler.conversations_table,
                IndexName="TenantUpdatedIndex",
                KeyConditionExpression="TenantId = :tid",
                ExpressionAttributeValues={":tid": {"S": tenant_id}},
                Select="COUNT"
            )
            total = count_response.get("Count", 0)

            return items, next_cursor, total

    @staticmethod
    async def update_conversation(
        tenant_id: str,
        conversation_id: str,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update a conversation"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            update_expr_parts = ["UpdatedAt = :now"]
            expr_values = {":now": {"S": datetime.utcnow().isoformat()}}

            if title is not None:
                update_expr_parts.append("Title = :title")
                expr_values[":title"] = {"S": title}

            if model is not None:
                update_expr_parts.append("Model = :model")
                expr_values[":model"] = {"S": model}

            if system_prompt is not None:
                update_expr_parts.append("SystemPrompt = :sp")
                expr_values[":sp"] = {"S": system_prompt}

            if metadata is not None:
                update_expr_parts.append("Metadata = :meta")
                expr_values[":meta"] = {"S": json.dumps(metadata)}

            await client.update_item(
                TableName=ConversationHandler.conversations_table,
                Key={
                    "TenantId": {"S": tenant_id},
                    "ConversationId": {"S": conversation_id}
                },
                UpdateExpression="SET " + ", ".join(update_expr_parts),
                ExpressionAttributeValues=expr_values
            )

        return await ConversationHandler.get_conversation(tenant_id, conversation_id)

    @staticmethod
    async def delete_conversation(tenant_id: str, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            # First delete all messages
            await ConversationHandler._delete_all_messages(conversation_id)

            # Then delete the conversation
            await client.delete_item(
                TableName=ConversationHandler.conversations_table,
                Key={
                    "TenantId": {"S": tenant_id},
                    "ConversationId": {"S": conversation_id}
                }
            )
            return True

    @staticmethod
    async def add_message(
        conversation_id: str,
        tenant_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """Add a message to a conversation"""
        now = datetime.utcnow()
        message_id = f"{now.strftime('%Y%m%d%H%M%S%f')}#{str(uuid.uuid4())[:8]}"

        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            item = {
                "ConversationId": {"S": conversation_id},
                "MessageId": {"S": message_id},
                "Role": {"S": role},
                "Content": {"S": content},
                "CreatedAt": {"S": now.isoformat()},
            }

            if model:
                item["Model"] = {"S": model}
            if provider:
                item["Provider"] = {"S": provider}
            if tokens_in is not None:
                item["TokensIn"] = {"N": str(tokens_in)}
            if tokens_out is not None:
                item["TokensOut"] = {"N": str(tokens_out)}
            if latency_ms is not None:
                item["LatencyMs"] = {"N": str(latency_ms)}
            if cost_usd is not None:
                item["CostUsd"] = {"N": str(cost_usd)}

            await client.put_item(
                TableName=ConversationHandler.messages_table,
                Item=item
            )

            # Update conversation's UpdatedAt and MessageCount
            await client.update_item(
                TableName=ConversationHandler.conversations_table,
                Key={
                    "TenantId": {"S": tenant_id},
                    "ConversationId": {"S": conversation_id}
                },
                UpdateExpression="SET UpdatedAt = :now ADD MessageCount :inc",
                ExpressionAttributeValues={
                    ":now": {"S": now.isoformat()},
                    ":inc": {"N": "1"}
                }
            )

        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now.isoformat(),
            "model": model,
            "provider": provider,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }

    @staticmethod
    async def get_messages(
        conversation_id: str,
        limit: int = 100,
        cursor: Optional[str] = None,
        order: str = "asc"
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Get messages for a conversation"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            query_kwargs = {
                "TableName": ConversationHandler.messages_table,
                "KeyConditionExpression": "ConversationId = :cid",
                "ExpressionAttributeValues": {
                    ":cid": {"S": conversation_id}
                },
                "ScanIndexForward": order == "asc",
                "Limit": limit
            }

            if cursor:
                try:
                    query_kwargs["ExclusiveStartKey"] = json.loads(
                        base64.b64decode(cursor).decode()
                    )
                except Exception:
                    pass

            response = await client.query(**query_kwargs)

            messages = [
                ConversationHandler._deserialize_message(item)
                for item in response.get("Items", [])
            ]

            next_cursor = None
            if "LastEvaluatedKey" in response:
                next_cursor = base64.b64encode(
                    json.dumps(response["LastEvaluatedKey"]).encode()
                ).decode()

            return messages, next_cursor

    @staticmethod
    async def get_all_messages_for_context(conversation_id: str) -> List[Dict[str, str]]:
        """Get all messages formatted for LLM context."""
        messages, _ = await ConversationHandler.get_messages(
            conversation_id,
            limit=1000,
            order="asc"
        )

        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

    @staticmethod
    def _deserialize_conversation(item: Dict) -> Dict[str, Any]:
        """Deserialize a DynamoDB conversation item"""
        result = {
            "conversation_id": item["ConversationId"]["S"],
            "tenant_id": item["TenantId"]["S"],
            "title": item["Title"]["S"],
            "model": item["Model"]["S"],
            "created_at": item["CreatedAt"]["S"],
            "updated_at": item["UpdatedAt"]["S"],
            "message_count": int(item["MessageCount"]["N"]),
        }

        if "SystemPrompt" in item:
            result["system_prompt"] = item["SystemPrompt"]["S"]

        if "Metadata" in item:
            try:
                result["metadata"] = json.loads(item["Metadata"]["S"])
            except Exception:
                result["metadata"] = None

        return result

    @staticmethod
    def _deserialize_message(item: Dict) -> Dict[str, Any]:
        """Deserialize a DynamoDB message item"""
        result = {
            "message_id": item["MessageId"]["S"],
            "conversation_id": item["ConversationId"]["S"],
            "role": item["Role"]["S"],
            "content": item["Content"]["S"],
            "created_at": item["CreatedAt"]["S"],
        }

        if "Model" in item:
            result["model"] = item["Model"]["S"]
        if "Provider" in item:
            result["provider"] = item["Provider"]["S"]
        if "TokensIn" in item:
            result["tokens_in"] = int(item["TokensIn"]["N"])
        if "TokensOut" in item:
            result["tokens_out"] = int(item["TokensOut"]["N"])
        if "LatencyMs" in item:
            result["latency_ms"] = float(item["LatencyMs"]["N"])
        if "CostUsd" in item:
            result["cost_usd"] = float(item["CostUsd"]["N"])

        return result

    @staticmethod
    async def _delete_all_messages(conversation_id: str):
        """Delete all messages for a conversation (batch delete)"""
        session = aioboto3.Session()
        async with session.client("dynamodb", region_name=ConversationHandler.region) as client:
            paginator_kwargs = {
                "TableName": ConversationHandler.messages_table,
                "KeyConditionExpression": "ConversationId = :cid",
                "ExpressionAttributeValues": {":cid": {"S": conversation_id}},
                "ProjectionExpression": "ConversationId, MessageId"
            }

            while True:
                response = await client.query(**paginator_kwargs)
                items = response.get("Items", [])

                if not items:
                    break

                # Batch delete (max 25 items per batch)
                for i in range(0, len(items), 25):
                    batch = items[i:i+25]
                    await client.batch_write_item(
                        RequestItems={
                            ConversationHandler.messages_table: [
                                {
                                    "DeleteRequest": {
                                        "Key": {
                                            "ConversationId": item["ConversationId"],
                                            "MessageId": item["MessageId"]
                                        }
                                    }
                                }
                                for item in batch
                            ]
                        }
                    )

                if "LastEvaluatedKey" not in response:
                    break
                paginator_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
