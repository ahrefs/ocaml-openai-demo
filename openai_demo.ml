open Ppx_yojson_conv_lib.Yojson_conv.Primitives

let api_key = Sys.getenv "OPENAI_API_KEY"

module OpenAI = struct
  let model = "gpt-4o"

  type json = Yojson.Safe.t

  let yojson_of_json x = x

  module Response = struct
    type response_message = { content : string option } [@@deriving of_yojson] [@@yojson.allow_extra_fields]

    type choice = { message : response_message } [@@deriving of_yojson] [@@yojson.allow_extra_fields]

    type response = { choices : choice list } [@@deriving of_yojson] [@@yojson.allow_extra_fields]
  end

  module Request = struct
    type message = {
      role : string;
      content : string;
    }
    [@@deriving yojson_of]
    type json_schema = {
      name : string;
      schema : json;
    }
    [@@deriving yojson_of]

    type response_format = {
      typ : string; [@key "type"]
      json_schema : json_schema;
    }
    [@@deriving yojson_of]

    type request = {
      model : string;
      messages : message list;
      response_format : response_format;
    }
    [@@deriving yojson_of]
  end

  let send ?(debug = false) request =
    let body = `Raw ("application/json", request |> Request.yojson_of_request |> Yojson.Safe.to_string) in
    let () =
      if debug then (
        let (`Raw (_, body)) = body in
        Printf.eprintf "Body: %s\n" body)
    in
    let headers = [ "Authorization: Bearer " ^ api_key ] in
    match Devkit.Web.http_request ~headers ~body `POST "https://api.openai.com/v1/chat/completions" with
    | `Error e -> Error e
    | `Ok response ->
    try Ok (response |> Yojson.Safe.from_string |> Response.response_of_yojson)
    with exn -> Error (Printf.sprintf "error while parsing the response %s: %S" (Printexc.to_string exn) response)
end

module Math_reasoning = struct
  type step = {
    explanation : string;
    output : float;
  }
  [@@deriving jsonschema, yojson]

  type math_reasoning = {
    steps : step list;
    final_answer : float;
  }
  [@@deriving jsonschema, yojson] [@@yojson.allow_extra_fields]
end

let math_tutor_request user_prompt =
  {
    OpenAI.Request.model = OpenAI.model;
    messages =
      [
        {
          role = "system";
          content =
            "You are a helpful math tutor. You will be provided with a math problem, and your goal will be to output a \
             step by step solution, along with a final answer. For each step, just provide the output as an equation \
             and use the explanation field to detail the reasoning.";
        };
        { role = "user"; content = user_prompt };
      ];
    response_format =
      {
        typ = "json_schema";
        json_schema = { name = "math_reasoning"; schema = Math_reasoning.math_reasoning_jsonschema };
      };
  }

let extract_steps { OpenAI.Response.message = { content }; _ } =
  match content with
  | None -> ()
  | Some content ->
  match content |> Yojson.Safe.from_string |> Math_reasoning.math_reasoning_of_yojson with
  | exception Ppx_yojson_conv_lib.Yojson_conv.Of_yojson_error (exn, json) ->
    Printf.eprintf "unable to parse response, error %s: %s\n" (Printexc.to_string exn) (Yojson.Safe.to_string json)
  | { Math_reasoning.steps; final_answer } ->
    List.iteri
      (fun i { Math_reasoning.explanation; output } ->
        Printf.printf "Step %d: %s\n" i explanation;
        Printf.printf "Output: %f\n" output)
      steps;
    Printf.printf "Final answer: %f\n" final_answer

let run user_prompt =
  let request = math_tutor_request user_prompt in
  match OpenAI.send request with
  | Error e -> Printf.eprintf "error: %s\n" e
  | Ok { OpenAI.Response.choices = []; _ } -> Printf.eprintf "no choices returned by OpenAI\n"
  | Ok { OpenAI.Response.choices; _ } -> List.iter extract_steps choices

let () =
  let user_prompt = Sys.argv.(1) in
  run user_prompt
